import sys
#sys.path.append('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/models/')
sys.path.append('./')
from imports import *
from search import *
from utils import *

from typing import Dict, List, Optional
from fairseq import utils

pad=0
sos=1
eos=2
unk=3

@torch.no_grad()
def generate(
    model, 
    src_batch, 
    voc, 
    search,
    max_len, 
    K = 2):
    """Generate translations. Match the api of other fairseq generators.
    Args:
        models (List[~fairseq.models.FairseqModel]): ensemble of models
        sample (dict): batch
        prefix_tokens (torch.LongTensor, optional): force decoder to begin
            with these tokens
        constraints (torch.LongTensor, optional): force decoder to include
            the list of constraints
        bos_token (int, optional): beginning of sentence token
            (default: eos)
    """

    # bsz: total number of sentences in beam
    # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
    src_tokens = src_batch.src
    bsz, src_len = src_tokens.size()[:2]
    beam_size = K
    
    # compute the encoder output for each beam
    encoder_outs = None
    memory = model.forward_encode(src_batch)
    if model.__class__.__name__ is 'TransformerSeq2Seq':
        encoder_outs = memory.expand(bsz * beam_size, memory.size()[1], memory.size()[2])
    elif model.__class__.__name__ is 'RNNSeq2Seq':
        encoder_outs = (memory[0].expand(bsz * beam_size, memory[0].size()[1], memory[0].size()[2]), 
                        memory[1].expand(memory[1].size()[0], bsz * beam_size, memory[1].size()[2]))
    else:
        # The cnn case
        encoder_outs = memory
    
    # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores

    # initialize buffers
    scores = (
        torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
    )  # +1 for eos; pad is never chosen for scoring
    tokens = (
        torch.zeros(bsz * beam_size, max_len + 2)
        .to(src_tokens)
        .long()
        .fill_(pad)
    )  # +2 for eos and pad
    tokens[:, 0] = sos

    # A list that indicates candidates that should be ignored.
    # For example, suppose we're sampling and have already finalized 2/5
    # samples. Then cands_to_ignore would mark 2 positions as being ignored,
    # so that we only finalize the remaining 3 samples.
    cands_to_ignore = (
        torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
    )  # forward and backward-compatible False mask

    # list of completed sentences
    finalized = torch.jit.annotate(
        List[List[Dict[str, Tensor]]],
        [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
    )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

    # a boolean array indicating if the sentence at the index is finished or not
    finished = [False for i in range(bsz)]
    num_remaining_sent = bsz  # number of sentences remaining

    # number of candidate hypos per step
    cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

    # offset arrays for converting between different indexing schemes
    bbsz_offsets = (
        (torch.arange(0, bsz) * beam_size)
        .unsqueeze(1)
        .type_as(tokens)
        .to(src_tokens.device)
    )
    cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

    reorder_state: Optional[Tensor] = None
    batch_idxs: Optional[Tensor] = None

    original_batch_idxs: Optional[Tensor] = None
    original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

    for step in range(max_len + 1):  # one extra step for EOS marker
        # Get the log probabilities based on previous outputs
        prev_output_tokens = tokens[:, : step + 1]
        #print(prev_output_tokens)
        lprobs = model.forward_decode(encoder_outs, src_batch, prev_output_tokens)

        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

        lprobs[:, pad] = -math.inf  # never select pad

        # Handle max length constraint
        if step >= max_len:
            lprobs[:, : eos] = -math.inf
            lprobs[:, eos + 1 :] = -math.inf

        scores = scores.type_as(lprobs)
        eos_bbsz_idx = torch.empty(0).to(
            tokens
        )  # indices of hypothesis ending with eos (finished sentences)
        eos_scores = torch.empty(0).to(
            scores
        )  # scores of hypothesis ending with eos (finished sentences)

        # Shape: (batch, cand_size)
        cand_scores, cand_indices, cand_beams = search.step(
            step,
            lprobs.view(bsz, -1, voc.num_words),
            scores.view(bsz, beam_size, -1)[:, :, :step],
            prev_output_tokens,
            original_batch_idxs,
        )

        # cand_bbsz_idx contains beam indices for the top candidate
        # hypotheses, with a range of values: [0, bsz*beam_size),
        # and dimensions: [bsz, cand_size]
        cand_bbsz_idx = cand_beams.add(bbsz_offsets)

        # finalize hypotheses that end in eos
        # Shape of eos_mask: (batch size, beam size)
        eos_mask = cand_indices.eq(eos) & cand_scores.ne(-math.inf)
        eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

        # only consider eos when it's among the top beam_size indices
        # Now we know what beam item(s) to finish
        # Shape: 1d list of absolute-numbered
        eos_bbsz_idx = torch.masked_select(
            cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
        )

        finalized_sents: List[int] = []
        if eos_bbsz_idx.numel() > 0:
            eos_scores = torch.masked_select(
                cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents = finalize_hypos(
                step,
                eos_bbsz_idx,
                eos_scores,
                tokens,
                scores,
                finalized,
                finished,
                beam_size,
                max_len,
            )
            num_remaining_sent -= len(finalized_sents)

        assert num_remaining_sent >= 0
        if num_remaining_sent == 0:
            break
        if step >= max_len:
            break
        assert step < max_len, f"{step} < {max_len}"

        # Remove finalized sentences (ones for which {beam_size}
        # finished hypotheses have been generated) from the batch.
        if len(finalized_sents) > 0:
            new_bsz = bsz - len(finalized_sents)

            # construct batch_idxs which holds indices of batches to keep for the next pass
            batch_mask = torch.ones(
                bsz, dtype=torch.bool, device=cand_indices.device
            )
            batch_mask[finalized_sents] = False
            # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
            batch_idxs = torch.arange(
                bsz, device=cand_indices.device
            ).masked_select(batch_mask)

            # Choose the subset of the hypothesized constraints that will continue
            search.prune_sentences(batch_idxs)

            eos_mask = eos_mask[batch_idxs]
            cand_beams = cand_beams[batch_idxs]
            bbsz_offsets.resize_(new_bsz, 1)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            cand_scores = cand_scores[batch_idxs]
            cand_indices = cand_indices[batch_idxs]

            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[batch_idxs]
            src_lengths = src_lengths[batch_idxs]
            cands_to_ignore = cands_to_ignore[batch_idxs]

            scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

            bsz = new_bsz
        else:
            batch_idxs = None

        # Set active_mask so that values > cand_size indicate eos hypos
        # and values < cand_size indicate candidate active hypos.
        # After, the min values per row are the top candidate active hypos

        # Rewrite the operator since the element wise or is not supported in torchscript.

        eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
        active_mask = torch.add(
            eos_mask.type_as(cand_offsets) * cand_size,
            cand_offsets[: eos_mask.size(1)],
        )

        # get the top beam_size active hypotheses, which are just
        # the hypos with the smallest values in active_mask.
        # {active_hypos} indicates which {beam_size} hypotheses
        # from the list of {2 * beam_size} candidates were
        # selected. Shapes: (batch size, beam size)
        new_cands_to_ignore, active_hypos = torch.topk(
            active_mask, k=beam_size, dim=1, largest=False
        )

        # update cands_to_ignore to ignore any finalized hypos.
        cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
        # Make sure there is at least one active item for each sentence in the batch.
        assert (~cands_to_ignore).any(dim=1).all()

        # update cands_to_ignore to ignore any finalized hypos

        # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
        # can be selected more than once).
        active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
        active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

        active_bbsz_idx = active_bbsz_idx.view(-1)
        active_scores = active_scores.view(-1)

        # copy tokens and scores for active hypotheses

        # Set the tokens for each beam (can select the same row more than once)
        tokens[:, : step + 1] = torch.index_select(
            tokens[:, : step + 1], dim=0, index=active_bbsz_idx
        )
        # Select the next token for each of them
        tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
            cand_indices, dim=1, index=active_hypos
        )
        if step > 0:
            scores[:, :step] = torch.index_select(
                scores[:, :step], dim=0, index=active_bbsz_idx
            )
        scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
            cand_scores, dim=1, index=active_hypos
        )

        # Update constraints based on which candidates were selected for the next beam
        search.update_constraints(active_hypos)

        # reorder incremental state in decoder
        reorder_state = active_bbsz_idx

    # sort by score descending
    for sent in range(len(finalized)):
        scores = torch.tensor(
            [float(elem["score"].item()) for elem in finalized[sent]]
        )
        _, sorted_scores_indices = torch.sort(scores, descending=True)
        finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
        finalized[sent] = torch.jit.annotate(
            List[Dict[str, Tensor]], finalized[sent]
        )
    return finalized[0]

def finalize_hypos(
    step: int,
    bbsz_idx,
    eos_scores,
    tokens,
    scores,
    finalized: List[List[Dict[str, Tensor]]],
    finished: List[bool],
    beam_size: int,
    max_len: int,
):
    """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
    A sentence is finalized when {beam_size} finished items have been collected for it.
    Returns number of sentences (not beam items) being finalized.
    These will be removed from the batch and not processed further.
    Args:
        bbsz_idx (Tensor):
    """
    assert bbsz_idx.numel() == eos_scores.numel()

    # clone relevant token and attention tensors.
    # tokens is (batch * beam, max_len). So the index_select
    # gets the newly EOS rows, then selects cols 1..{step + 2}
    tokens_clone = tokens.index_select(0, bbsz_idx)[
        :, 1 : step + 2
    ]  # skip the first index, which is EOS

    tokens_clone[:, step] = eos

    # compute scores per token position
    pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
    pos_scores[:, step] = eos_scores
    # convert from cumulative to per-position scores
    pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

    # normalize sentence-level scores
    eos_scores /= (step + 1) ** 1.0 # length penalty = 1.0

    # cum_unfin records which sentences in the batch are finished.
    # It helps match indexing between (a) the original sentences
    # in the batch and (b) the current, possibly-reduced set of
    # sentences.
    cum_unfin: List[int] = []
    prev = 0
    for f in finished:
        if f:
            prev += 1
        else:
            cum_unfin.append(prev)

    # The keys here are of the form "{sent}_{unfin_idx}", where
    # "unfin_idx" is the index in the current (possibly reduced)
    # list of sentences, and "sent" is the index in the original,
    # unreduced batch
    # set() is not supported in script export
    sents_seen: Dict[str, Optional[Tensor]] = {}

    # For every finished beam item
    for i in range(bbsz_idx.size()[0]):
        idx = bbsz_idx[i]
        score = eos_scores[i]
        # sentence index in the current (possibly reduced) batch
        unfin_idx = idx // beam_size
        # sentence index in the original (unreduced) batch
        sent = unfin_idx + cum_unfin[unfin_idx]
        # Cannot create dict for key type '(int, int)' in torchscript.
        # The workaround is to cast int to string
        seen = str(sent.item()) + "_" + str(unfin_idx.item())
        if seen not in sents_seen:
            sents_seen[seen] = None

        # An input sentence (among those in a batch) is finished when
        # beam_size hypotheses have been collected for it
        if len(finalized[sent]) < beam_size:
            hypo_attn = torch.empty(0)

            finalized[sent].append(
                {
                    "tokens": tokens_clone[i],
                    "score": score,
                    "attention": hypo_attn,  # src_len x tgt_len
                    "alignment": torch.empty(0),
                    "positional_scores": pos_scores[i],
                }
            )

    newly_finished: List[int] = []

    for seen in sents_seen.keys():
        # check termination conditions for this sentence
        sent: int = int(float(seen.split("_")[0]))
        unfin_idx: int = int(float(seen.split("_")[1]))

        if not finished[sent] and is_finished(
            step, unfin_idx, max_len, len(finalized[sent]), beam_size
        ):
            finished[sent] = True
            newly_finished.append(unfin_idx)

    return newly_finished

def is_finished(
    step: int,
    unfin_idx: int,
    max_len: int,
    finalized_sent_len: int,
    beam_size: int,
):
    """
    Check whether decoding for a sentence is finished, which
    occurs when the list of finalized sentences has reached the
    beam size, or when we reach the maximum length.
    """
    assert finalized_sent_len <= beam_size
    if finalized_sent_len == beam_size or step == max_len:
        return True
    return False

def generate_topK_beams(data_iter, model, voc, max_len=100, K=2, bs_strategy=None):
    """
    Predict the next query of a list of queries. Assumes batch size of 1.
    :param pairs: a list of query pairs
    :param voc: voc from the training data set
    :param bs_strategy (Search): beam search strategy
    :param model: trained model
    :return targets (list): a list of target queries. Each query is a list of tokens
    :return preds_list (list): a list of predicted queries. Each query is a dict (k:token, v:score)
    """

    model.eval()
    
    #data_iter = list(dataloader.data_to_batch(voc, pairs, device, batch_size=1)) #len(pairs)
    targets = []
    preds_list = []

    for i, batch in enumerate(data_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos else src
        trg = trg[:-1] if trg[-1] == eos else trg

        target = [voc.index2word[token.item()] for token in trg]
        # remove <sem> (if it is there)
        target = target[1:] if target[0].lower() == 'sem' else target
        targets.append(target)
        
        # BS when bs_strategy is not None, otherwise greedy decoding
        if bs_strategy is None:
            result = greedy_decode(model, batch, voc, max_len)
        else: 
            result = generate(model, batch, voc, bs_strategy, max_len, K)
        
        preds_list.append(result)

    return targets, preds_list

@torch.no_grad()
def greedy_decode(model, src_batch, voc, max_len=100):
    src_tokens = src_batch.src
    bsz, src_len = src_tokens.size()[:2]
    beam_size = 1
    
    # compute the encoder output for each beam
    encoder_outs = None
    memory = model.forward_encode(src_batch)
    encoder_outs = memory
    
    # initialize buffers
    ys = torch.ones(1, 1).fill_(sos).type_as(src_tokens.data)
    output = []
    for step in range(max_len + 1):  # one extra step for EOS marker
        # Get the log probabilities based on previous outputs
        lprobs = model.forward_decode(memory, src_batch, ys)
        #lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
        #lprobs[:, pad] = -math.inf  # never select pad
        
        _, next_word = torch.max(lprobs, dim = 1)
        next_word = next_word.data.item()
        
        # Update buffer and output
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tokens.data).fill_(next_word)], dim=1)
        output.append(next_word)
        
    output = np.array(output)
    # cut off everything starting from </s> 
    first_eos = np.where(output==eos)[0]
    if len(first_eos) > 0:
        output = output[:first_eos[0]]
    
    pred = [voc.index2word[token.item()] for token in output]
    '''
    # Remove the sentence-level token
    # Note: 'sem' will get ignored later in evaluation 
    #       removing it here causes errors
    if len(pred) > 1:
        pred = pred[1:] if pred[0].lower() == 'sem' else pred
    elif len(pred) == 1:
        pred = [] if pred[0].lower() == 'sem' else pred
    '''
    return pred

if __name__ == "__main__":
    from dataloader import *
    from transformer import core, training

    default_tokens = training.DEFAULT_TOKENS
    max_length = training.MAX_LENGTH

    # Use CUDA if it is available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = 'F://data//processed//sdss//model_data//sampled//'
    save_dir = os.path.join("data", "save")
    datafile = os.path.join(data_dir, 'train.txt')
    voc, pairs = load_query_pairs(default_tokens, data_dir, 'sdss', datafile, save_dir, max_length)
    pairs = trimRareWords(voc, pairs, 3)

    datafile = os.path.join(data_dir, 'test.txt')
    _, pairs = load_query_pairs(default_tokens, data_dir, 'sdss', datafile, save_dir, max_length)
    pairs = trimRareWords(voc, pairs, 3)
    test_data = pairs[:2]

    # Load model
    model_pth = '../../saved_models/sdss/transformer_pred_sdss.pth'
    loaded_model = core.make_model(voc.num_words, voc.num_words, N=3, d_model=128, d_ff=512, h=8, dropout=0.1).to(device)
    loaded_model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu'))) # Load to cpu
    loaded_model.eval()
    print("Model loaded...")

    data_iter = list(training.data_to_batch(voc, test_data, device, batch_size=1)) #len(pairs)
    targets = []
    preds_list = []

    for i, batch in enumerate(data_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos else src
        trg = trg[:-1] if trg[-1] == eos else trg      
      
        result = generate(loaded_model, batch, voc, BeamSearch(voc), 100, 2)
        #result = decoding.beam_decode(loaded_model, batch.src, batch.src_mask, max_len=max_len)

        #print(result)