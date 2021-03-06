TABLES = ['apogeeDesign',
          'apogeeField',
          'apogeeObject',
          'apogeePlate',
          'apogeeStar',
          'apogeeStarAllVisit',
          'apogeeStarVisit',
          'apogeeVisit',
          'aspcapStar',
          'aspcapStarCovar',
          'AtlasOutline',
          'cannonStar',
          'DataConstants',
          'DBColumns',
          'DBObjects',
          'DBViewCols',
          'Dependency',
          'detectionIndex',
          'Diagnostics',
          'emissionLinesPort',
          'Field',
          'FieldProfile',
          'FIRST',
          'Frame',
          'galSpecExtra',
          'galSpecIndx',
          'galSpecInfo',
          'galSpecLine',
          'HalfSpace',
          'History',
          'Inventory',
          'LoadHistory',
          'mangaDAPall',
          'mangaDRPall',
          'mangaFirefly',
          'mangaHIall',
          'mangaHIbonus',
          'mangaPipe3D',
          'mangatarget',
          'marvelsStar',
          'marvelsVelocityCurveUF1D',
          'Mask',
          'MaskedObject',
          'mastar_goodstars',
          'mastar_goodvisits',
          'Neighbors',
          'nsatlas',
          'PartitionMap',
          'PhotoObjAll',
          'PhotoObjDR7',
          'PhotoPrimaryDR7',
          'PhotoProfile',
          'Photoz',
          'PhotozErrorMap',
          'Plate2Target',
          'PlateX',
          'ProfileDefs',
          'ProperMotions',
          'qsoVarPTF',
          'qsoVarStripe',
          'RC3',
          'RecentQueries',
          'Region',
          'Region2Box',
          'RegionArcs',
          'RegionPatch',
          'RegionTypes',
          'Rmatrix',
          'ROSAT',
          'Run',
          'RunShift',
          'sdssBestTarget2Sector',
          'SDSSConstants',
          'sdssEbossFirefly',
          'sdssImagingHalfSpaces',
          'sdssPolygon2Field',
          'sdssPolygons',
          'sdssSector',
          'sdssSector2Tile',
          'sdssTargetParam',
          'sdssTileAll',
          'sdssTiledTargetAll',
          'sdssTilingGeometry',
          'sdssTilingInfo',
          'sdssTilingRun',
          'segueTargetAll',
          'SiteConstants',
          'SiteDBs',
          'SiteDiagnostics',
          'SpecDR7',
          'SpecObjAll',
          'SpecPhotoAll',
          'spiders_quasar',
          'sppLines',
          'sppParams',
          'sppTargets',
          'stellarMassFSPSGranEarlyDust',
          'stellarMassFSPSGranEarlyNoDust',
          'stellarMassFSPSGranWideDust',
          'stellarMassFSPSGranWideNoDust',
          'stellarMassPassivePort',
          'stellarMassPCAWiscBC03',
          'stellarMassPCAWiscM11',
          'stellarMassStarformingPort',
          'StripeDefs',
          'Target',
          'TargetInfo',
          'thingIndex',
          'TwoMass',
          'TwoMassXSC',
          'USNO',
          'Versions',
          'WISE_allsky',
          'WISE_xmatch',
          'wiseForcedTarget',
          'Zone',
          'zoo2MainPhotoz',
          'zoo2MainSpecz',
          'zoo2Stripe82Coadd1',
          'zoo2Stripe82Coadd2',
          'zoo2Stripe82Normal',
          'zooConfidence',
          'zooMirrorBias',
          'zooMonochromeBias',
          'zooNoSpec',
          'zooSpec',
          'zooVotes']
VIEWS = ['AncillaryTarget1',
	'AncillaryTarget2',
	'ApogeeAspcapFlag',
	'ApogeeExtraTarg',
	'ApogeeParamFlag',
	'ApogeeStarFlag',
	'ApogeeTarget1',
	'ApogeeTarget2',
	'BossTarget1',
	'CalibStatus',
	'Columns',
	'CoordType',
	'eBossTarget0',
	'FieldQuality',
	'FramesStatus',
	'Galaxy',
	'GalaxyTag',
	'HoleType',
	'ImageStatus',
	'InsideMask',
	'MaskType',
	'PhotoFamily',
	'PhotoFlags',
	'PhotoMode',
	'PhotoObj',
	'PhotoPrimary',
	'PhotoSecondary',
	'PhotoStatus',
	'PhotoTag',
	'PhotoType',
	'PrimTarget',
	'ProgramType',
	'PspStatus',
	'RegionConvex',
	'ResolveStatus',
	'sdssTile',
	'sdssTilingBoundary',
    'sdssTilingMask',
	'SecTarget',
	'segue1SpecObjAll',
	'Segue1Target1',
	'Segue1Target2',
	'segue2SpecObjAll',
	'Segue2Target1',
	'Segue2Target2',
	'segueSpecObjAll',
	'Sky',
	'SourceType',
	'SpecialTarget1',
    'SpecialTarget2',
	'SpecObj',
	'SpecPhoto',
	'SpecPixMask',
	'SpecZWarning',
	'Star',
	'StarTag',
	'TableDesc',
	'TiMask',
	'Unknown'
]
FUNCTIONS = [
    'fAncillaryTarget1',
    'fAncillaryTarget1N',
    'fAncillaryTarget2',
    'fAncillaryTarget2N',
    'fApogeeAspcapFlag',
    'fApogeeAspcapFlagN',
    'fApogeeExtraTarg',
    'fApogeeExtraTargN',
    'fApogeeParamFlag',
    'fApogeeParamFlagN',
    'fApogeeStarFlag',
    'fApogeeStarFlagN',
    'fApogeeTarget1',
    'fApogeeTarget1N',
    'fApogeeTarget2',
    'fApogeeTarget2N',
    'fAspcapElemErrs',
    'fAspcapElemFlags',
    'fAspcapElems',
    'fAspcapElemsAll',
    'fAspcapFelemErrs',
    'fAspcapFelems',
    'fAspcapFelemsAll',
    'fAspcapParamErrs',
    'fAspcapParamFlags',
    'fAspcapParams',
    'fAspcapParamsAll',
    'fBossTarget1',
    'fBossTarget1N',
    'fCalibStatus',
    'fCalibStatusN',
    'fCamcol',
    'fCoordsFromEq',
    'fCoordType',
    'fCoordTypeN',
    'fCosmoAbsMag',
    'fCosmoAgeOfUniverse',
    'fCosmoComovDist2ObjectsRADEC',
    'fCosmoComovDist2ObjectsXYZ',
    'fCosmoComovingVolume',
    'fCosmoDa',
    'fCosmoDc',
    'fCosmoDistanceModulus',
    'fCosmoDl',
    'fCosmoDm',
    'fCosmoHubbleDistance',
    'fCosmoLookBackTime',
    'fCosmoQuantities',
    'fCosmoTimeInterval',
    'fCosmoZfromAgeOfUniverse',
    'fCosmoZfromDa',
    'fCosmoZfromDc',
    'fCosmoZfromDl',
    'fCosmoZfromDm',
    'fCosmoZfromLookBackTime',
    'fDatediffSec',
    'fDistanceArcMinEq',
    'fDistanceArcMinXYZ',
    'fDistanceEq',
    'fDistanceXyz',
    'fDMS',
    'fDMSbase',
    'fDocColumns',
    'fDocColumnsWithRank',
    'fDocFunctionParams',
    'fEbossTarget0',
    'fEbossTarget0N',
    'fEnum',
    'fEqFromMuNu',
    'fEtaFromEq',
    'fEtaToNormal',
    'fFiber',
    'fField',
    'fFieldMask',
    'fFieldMaskN',
    'fFieldQuality',
    'fFieldQualityN',
    'fFirstFieldBit',
    'fFootprintEq',
    'fFramesStatus',
    'fFramesStatusN',
    'fGetAlpha',
    'fGetBlob',
    'fGetLat',
    'fGetLon',
    'fGetLonLat',
    'fGetNearbyApogeeStarEq',
    'fGetNearbyFrameEq',
    'fGetNearbyMangaObjEq',
    'fGetNearbyMaStarObjEq',
    'fGetNearbyObjAllEq',
    'fGetNearbyObjAllXYZ',
    'fGetNearbyObjEq',
    'fGetNearbyObjXYZ',
    'fGetNearbySpecObjAllEq',
    'fGetNearbySpecObjAllXYZ',
    'fGetNearbySpecObjEq',
    'fGetNearbySpecObjXYZ',
    'fGetNearestApogeeStarEq',
    'fGetNearestFrameEq',
    'fGetNearestFrameidEq',
    'fGetNearestMangaObjEq',
    'fGetNearestMastarObjEq',
    'fGetNearestObjAllEq',
    'fGetNearestObjEq',
    'fGetNearestObjIdAllEq',
    'fGetNearestObjIdEq',
    'fGetNearestObjIdEqMode',
    'fGetNearestObjIdEqType',
    'fGetNearestObjXYZ',
    'fGetNearestSpecObjAllEq',
    'fGetNearestSpecObjAllXYZ',
    'fGetNearestSpecObjEq',
    'fGetNearestSpecObjIdAllEq',
    'fGetNearestSpecObjIdEq',
    'fGetNearestSpecObjXYZ',
    'fGetObjectsEq',
    'fGetObjectsMaskEq',
    'fGetObjFromRect',
    'fGetObjFromRectEq',
    'fGetUrlExpEq',
    'fGetUrlExpId',
    'fGetUrlFitsAtlas',
    'fGetUrlFitsBin',
    'fGetUrlFitsCFrame',
    'fGetUrlFitsField',
    'fGetUrlFitsMask',
    'fGetUrlFitsPlate',
    'fGetUrlFitsSpectrum',
    'fGetUrlFrameImg',
    'fGetUrlMangaCube',
    'fGetUrlNavEq',
    'fGetUrlNavId',
    'fGetUrlSpecImg',
    'fGetWCS',
    'fHMS',
    'fHMSbase',
    'fHoleType',
    'fHoleTypeN',
    'fHtmCoverBinaryAdvanced',
    'fHtmCoverCircleEq',
    'fHtmCoverCircleXyz',
    'fHtmCoverRegion',
    'fHtmCoverRegionAdvanced',
    'fHtmCoverRegionError',
    'fHtmEq',
    'fHtmEqToXyz',
    'fHtmGetCenterPoint',
    'fHtmGetCornerPoints',
    'fHtmGetString',
    'fHtmVersion',
    'fHtmXyz',
    'fHtmXyzToEq',
    'fIAUFromEq',
    'fImageStatus',
    'fImageStatusN',
    'fInFootprintEq',
    'fInsideMask',
    'fInsideMaskN',
    'fIsNumbers',
    'fLambdaFromEq',
    'fMagToFlux',
    'fMagToFluxErr',
    'fMaskType',
    'fMaskTypeN',
    'fMathGetBin',
    'fMJD',
    'fMJDToGMT',
    'fMuFromEq',
    'fMuNuFromEq',
    'fNormalizeString',
    'fNuFromEq',
    'fObj',
    'fObjID',
    'fObjidFromSDSS',
    'fObjidFromSDSSWithFF',
    'fPhotoDescription',
    'fPhotoFlags',
    'fPhotoFlagsN',
    'fPhotoMode',
    'fPhotoModeN',
    'fPhotoStatus',
    'fPhotoStatusN',
    'fPhotoType',
    'fPhotoTypeN',
    'fPlate',
    'fPolygonsContainingPointEq',
    'fPolygonsContainingPointXYZ',
    'fPrimaryObjID',
    'fPrimTarget',
    'fPrimTargetN',
    'fProgramType',
    'fProgramTypeN',
    'fPspStatus',
    'fPspStatusN',
    'fRegionContainsPointEq',
    'fRegionContainsPointXYZ',
    'fRegionFuzz',
    'fRegionGetObjectsFromRegionId',
    'fRegionGetObjectsFromString',
    'fRegionOverlapId',
    'fRegionsContainingPointEq',
    'fRegionsContainingPointXYZ',
    'fRegionsIntersectingBinary',
    'fRegionsIntersectingString',
    'fReplace',
    'fReplaceMax',
    'fRerun',
    'fResolveStatus',
    'fResolveStatusN',
    'fRotateV3',
    'fRun',
    'fSDSS',
    'fSDSSfromObjID',
    'fSDSSfromSpecID',
    'fSecTarget',
    'fSecTargetN',
    'fSegue1Target1',
    'fSegue1Target1N',
    'fSegue1Target2',
    'fSegue1Target2N',
    'fSegue2Target1',
    'fSegue2Target1N',
    'fSegue2Target2',
    'fSegue2Target2N',
    'fSkyVersion',
    'fSourceType',
    'fSourceTypeN',
    'fSpecDescription',
    'fSpecialTarget1',
    'fSpecialTarget1N',
    'fSpecialTarget2',
    'fSpecialTarget2N',
    'fSpecidFromSDSS',
    'fSpecPixMask',
    'fSpecPixMaskN',
    'fSpecZWarning',
    'fSpecZWarningN',
    'fStripeOfRun',
    'fStripeToNormal',
    'fStripOfRun',
    'fTiMask',
    'fTiMaskN',
    'fTokenAdvance',
    'fTokenNext',
    'fTokenStringToTable',
    'fVarBinToHex',
    'fWedgeV3'
]

TIMEOUTLIST = [
    'select min(top50.psfMag_r) as psfMagMed_r from ( select top 50 percent psfMag_r from photoObj as obj order by psfMag_r desc ) as top50',
    'select count(*) from specphoto s, phototag p where s.objid=p.objid and p.specobjid=0',
    "select * from dbo.fDocColumns('PhotoObj')",
    "SELECT TOP 50 p.run,p.rerun,p.camCol,p.field,p.obj, 'ugriz' as filter FROM dbo.fGetNearbyObjEq(241.771530,53.599601,0.5) as b, BESTDR2..PhotoObj as p WHERE b.objID = p.objID AND ( p.type = 3 OR p.type = 6)",
    "SELECT G.objID, profile.band, profile.bin, profile.profMean FROM Galaxy G,PhotoProfile profile WHERE (G.ra > 300) and G.dec between -3 and 4 and (G.petroMag_r - G.extinction_r < 20.5) and (G.petroMag_r - G.extinction_r >= 20.0) and (G.flags & dbo.fPhotoFlags('BINNED1')) > 0 and (G.flags & ( dbo.fPhotoFlags('BLENDED') + dbo.fPhotoFlags('NODEBLEND') + dbo.fPhotoFlags('CHILD'))) != dbo.fPhotoFlags('BLENDED') and (G.flags & (dbo.fPhotoFlags('SATURATED'))) = 0 and (G.objID = profile.objID) and (profile.band != 0) and (profile.band != 4) and (profile.bin < 3)",
    "SELECT p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z, p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z, p.fieldid, good = -(dbo.fPhotoFlags('BRIGHT') + dbo.fPhotoFlags('EDGE') + dbo.fPhotoFlags('BLENDED') + dbo.fPhotoFlags('COSMIC_RAY') + dbo.fPhotoFlags('SATURATED') )+ 266255 FROM PhotoObj as p WHERE p.ra between 270 and 360 AND p.dec between -10 and 0 AND p.psfMag_r < 21.5 AND p.psfMag_i < 21.5",
    "SELECT p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z, p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z,f.mjd_u,f.mjd_g,f.mjd_r,f.mjd_i,f.mjd_z FROM PhotoObj as p inner join field as f on f.fieldid=p.fieldid WHERE p.psfMag_r < 21 AND p.psfMag_i < 21 AND (p.flags & (dbo.fPhotoFlags('BRIGHT') + dbo.fPhotoFlags('EDGE') + dbo.fPhotoFlags('BLENDED') + dbo.fPhotoFlags('COSMIC_RAY') + dbo.fPhotoFlags('SATURATED')) = 0) AND (abs(119.648439 - p.ra) < 0.27) AND (abs(21.964472 - p.dec) < 0.5)",
    "SELECT p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z, p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z,f.mjd_u,f.mjd_g,f.mjd_r,f.mjd_i,f.mjd_z INTO MyDB.obj1568 FROM PhotoObj as p inner join field as f on f.fieldid=p.fieldid WHERE p.psfMag_r < 21 AND p.psfMag_i < 21 AND (p.flags & (dbo.fPhotoFlags('BRIGHT') + dbo.fPhotoFlags('EDGE') + dbo.fPhotoFlags('BLENDED') + dbo.fPhotoFlags('COSMIC_RAY') + dbo.fPhotoFlags('SATURATED')) = 0) AND (abs(333.563171 - p.ra) < 0.27) AND (abs(12.370988 - p.dec) < 0.5)",
    "SELECT p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z, p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z,f.mjd_u,f.mjd_g,f.mjd_r,f.mjd_i,f.mjd_z INTO MyDB.obj1719 FROM PhotoObj as p inner join field as f on f.fieldid=p.fieldid WHERE p.psfMag_r < 21 AND p.psfMag_i < 21 AND (p.flags & (dbo.fPhotoFlags('BRIGHT') + dbo.fPhotoFlags('EDGE') + dbo.fPhotoFlags('BLENDED') + dbo.fPhotoFlags('COSMIC_RAY') + dbo.fPhotoFlags('SATURATED')) = 0) AND (abs(336.316319 - p.ra) < 0.27) AND (abs(-0.097641 - p.dec) < 0.5)",
    "SELECT p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z, p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z,f.mjd_u,f.mjd_g,f.mjd_r,f.mjd_i,f.mjd_z INTO MyDB.obj1824 FROM PhotoObj as p inner join field as f on f.fieldid=p.fieldid WHERE p.psfMag_r < 21 AND p.psfMag_i < 21 AND (p.flags & (dbo.fPhotoFlags('BRIGHT') + dbo.fPhotoFlags('EDGE') + dbo.fPhotoFlags('BLENDED') + dbo.fPhotoFlags('COSMIC_RAY') + dbo.fPhotoFlags('SATURATED')) = 0) AND (abs(338.131579 - p.ra) < 0.27) AND (abs(-0.502541 - p.dec) < 0.5)",
    'Select objID "SDSS_objID",skyVersion "SDSS_skyVersion",field "SDSS_field",mode "SDSS_mode",nChild "SDSS_nChild",type "SDSS_type",probPSF "SDSS_probPSF",insideMask "SDSS_insideMask",flags "SDSS_flags",rowc "SDSS_rowc",rowcErr "SDSS_rowcErr",colc "SDSS_colc",colcErr "SDSS_colcErr",rowv "SDSS_rowv",rowvErr "SDSS_rowvErr",colv "SDSS_colv",colvErr "SDSS_colvErr",sky_u "SDSS_sky_u",sky_g "SDSS_sky_g",sky_r "SDSS_sky_r",sky_i "SDSS_sky_i",sky_z "SDSS_sky_z",skyErr_u "SDSS_skyErr_u",skyErr_g "SDSS_skyErr_g",skyErr_r "SDSS_skyErr_r",skyErr_i "SDSS_skyErr_i",skyErr_z "SDSS_skyErr_z",psfMag_u "SDSS_psfMag_u",psfMag_g "SDSS_psfMag_g",psfMag_r "SDSS_psfMag_r",psfMag_i "SDSS_psfMag_i",psfMag_z "SDSS_psfMag_z",psfMagErr_u "SDSS_psfMagErr_u",psfMagErr_g "SDSS_psfMagErr_g",psfMagErr_r "SDSS_psfMagErr_r",psfMagErr_i "SDSS_psfMagErr_i",psfMagErr_z "SDSS_psfMagErr_z",fiberMag_u "SDSS_fiberMag_u",fiberMag_g "SDSS_fiberMag_g",fiberMag_r "SDSS_fiberMag_r",fiberMag_i "SDSS_fiberMag_i",fiberMag_z "SDSS_fiberMag_z",fiberMagErr_u "SDSS_fiberMagErr_u",fiberMagErr_g "SDSS_fiberMagErr_g",fiberMagErr_r "SDSS_fiberMagErr_r",fiberMagErr_i "SDSS_fiberMagErr_i",fiberMagErr_z "SDSS_fiberMagErr_z",petroMag_u "SDSS_petroMag_u",petroMag_g "SDSS_petroMag_g",petroMag_r "SDSS_petroMag_r",petroMag_i "SDSS_petroMag_i",petroMag_z "SDSS_petroMag_z",petroMagErr_u "SDSS_petroMagErr_u",petroMagErr_g "SDSS_petroMagErr_g",petroMagErr_r "SDSS_petroMagErr_r",petroMagErr_i "SDSS_petroMagErr_i",petroMagErr_z "SDSS_petroMagErr_z",petroRad_u "SDSS_petroRad_u",petroRad_g "SDSS_petroRad_g",petroRad_r "SDSS_petroRad_r",petroRad_i "SDSS_petroRad_i",petroRad_z "SDSS_petroRad_z",petroRadErr_u "SDSS_petroRadErr_u",petroRadErr_g "SDSS_petroRadErr_g",petroRadErr_r "SDSS_petroRadErr_r",petroRadErr_i "SDSS_petroRadErr_i",petroRadErr_z "SDSS_petroRadErr_z",petroR50_u "SDSS_petroR50_u",petroR50_g "SDSS_petroR50_g",petroR50_r "SDSS_petroR50_r",petroR50_i "SDSS_petroR50_i",petroR50_z "SDSS_petroR50_z",petroR50Err_u "SDSS_petroR50Err_u",petroR50Err_g "SDSS_petroR50Err_g",petroR50Err_r "SDSS_petroR50Err_r",petroR50Err_i "SDSS_petroR50Err_i",petroR50Err_z "SDSS_petroR50Err_z",petroR90_u "SDSS_petroR90_u",petroR90_g "SDSS_petroR90_g",petroR90_r "SDSS_petroR90_r",petroR90_i "SDSS_petroR90_i",petroR90_z "SDSS_petroR90_z",petroR90Err_u "SDSS_petroR90Err_u",petroR90Err_g "SDSS_petroR90Err_g",petroR90Err_r "SDSS_petroR90Err_r",petroR90Err_i "SDSS_petroR90Err_i",petroR90Err_z "SDSS_petroR90Err_z",isoRowc_u "SDSS_isoRowc_u",isoRowc_g "SDSS_isoRowc_g",isoRowc_r "SDSS_isoRowc_r",isoRowc_i "SDSS_isoRowc_i",isoRowc_z "SDSS_isoRowc_z",isoRowcErr_u "SDSS_isoRowcErr_u",isoRowcErr_g "SDSS_isoRowcErr_g",isoRowcErr_r "SDSS_isoRowcErr_r",isoRowcErr_i "SDSS_isoRowcErr_i",isoRowcErr_z "SDSS_isoRowcErr_z",isoRowcGrad_u "SDSS_isoRowcGrad_u",isoRowcGrad_g "SDSS_isoRowcGrad_g",isoRowcGrad_r "SDSS_isoRowcGrad_r",isoRowcGrad_i "SDSS_isoRowcGrad_i",isoRowcGrad_z "SDSS_isoRowcGrad_z",isoColc_u "SDSS_isoColc_u",isoColc_g "SDSS_isoColc_g",isoColc_r "SDSS_isoColc_r",isoColc_i "SDSS_isoColc_i",isoColc_z "SDSS_isoColc_z",isoColcErr_u "SDSS_isoColcErr_u",isoColcErr_g "SDSS_isoColcErr_g",isoColcErr_r "SDSS_isoColcErr_r",isoColcErr_i "SDSS_isoColcErr_i",isoColcErr_z "SDSS_isoColcErr_z",isoColcGrad_u "SDSS_isoColcGrad_u",isoColcGrad_g "SDSS_isoColcGrad_g",isoColcGrad_r "SDSS_isoColcGrad_r",isoColcGrad_i "SDSS_isoColcGrad_i",isoColcGrad_z "SDSS_isoColcGrad_z",isoA_u "SDSS_isoA_u",isoA_g "SDSS_isoA_g",isoA_r "SDSS_isoA_r",isoA_i "SDSS_isoA_i",isoA_z "SDSS_isoA_z",isoAErr_u "SDSS_isoAErr_u",isoAErr_g "SDSS_isoAErr_g",isoAErr_r "SDSS_isoAErr_r",isoAErr_i "SDSS_isoAErr_i",isoAErr_z "SDSS_isoAErr_z",isoB_u "SDSS_isoB_u",isoB_g "SDSS_isoB_g",isoB_r "SDSS_isoB_r",isoB_i "SDSS_isoB_i",isoB_z "SDSS_isoB_z",isoBErr_u "SDSS_isoBErr_u",isoBErr_g "SDSS_isoBErr_g",isoBErr_r "SDSS_isoBErr_r",isoBErr_i "SDSS_isoBErr_i",isoBErr_z "SDSS_isoBErr_z",isoAGrad_u "SDSS_isoAGrad_u",isoAGrad_g "SDSS_isoAGrad_g",isoAGrad_r "SDSS_isoAGrad_r",isoAGrad_i "SDSS_isoAGrad_i",isoAGrad_z "SDSS_isoAGrad_z",isoBGrad_u "SDSS_isoBGrad_u",isoBGrad_g "SDSS_isoBGrad_g",isoBGrad_r "SDSS_isoBGrad_r",isoBGrad_i "SDSS_isoBGrad_i",isoBGrad_z "SDSS_isoBGrad_z",isoPhi_u "SDSS_isoPhi_u",isoPhi_g "SDSS_isoPhi_g",isoPhi_r "SDSS_isoPhi_r",isoPhi_i "SDSS_isoPhi_i",isoPhi_z "SDSS_isoPhi_z",isoPhiErr_u "SDSS_isoPhiErr_u",isoPhiErr_g "SDSS_isoPhiErr_g",isoPhiErr_r "SDSS_isoPhiErr_r",isoPhiErr_i "SDSS_isoPhiErr_i",isoPhiErr_z "SDSS_isoPhiErr_z",isoPhiGrad_u "SDSS_isoPhiGrad_u",isoPhiGrad_g "SDSS_isoPhiGrad_g",isoPhiGrad_r "SDSS_isoPhiGrad_r",isoPhiGrad_i "SDSS_isoPhiGrad_i",isoPhiGrad_z "SDSS_isoPhiGrad_z",texture_u "SDSS_texture_u",texture_g "SDSS_texture_g",texture_r "SDSS_texture_r",texture_i "SDSS_texture_i",texture_z "SDSS_texture_z",lnLStar_u "SDSS_lnLStar_u",lnLStar_g "SDSS_lnLStar_g",lnLStar_r "SDSS_lnLStar_r",lnLStar_i "SDSS_lnLStar_i",lnLStar_z "SDSS_lnLStar_z",flags_u "SDSS_flags_u",flags_g "SDSS_flags_g",flags_r "SDSS_flags_r",flags_i "SDSS_flags_i",flags_z "SDSS_flags_z",type_u "SDSS_type_u",type_g "SDSS_type_g",type_r "SDSS_type_r",type_i "SDSS_type_i",type_z "SDSS_type_z",probPSF_u "SDSS_probPSF_u",probPSF_g "SDSS_probPSF_g",probPSF_r "SDSS_probPSF_r",probPSF_i "SDSS_probPSF_i",probPSF_z "SDSS_probPSF_z",status "SDSS_status",ra "RA",dec "DEC",offsetRa_u "SDSS_offsetRa_u",offsetRa_g "SDSS_offsetRa_g",offsetRa_r "SDSS_offsetRa_r",offsetRa_i "SDSS_offsetRa_i",offsetRa_z "SDSS_offsetRa_z",offsetDec_u "SDSS_offsetDec_u",offsetDec_g "SDSS_offsetDec_g",offsetDec_r "SDSS_offsetDec_r",offsetDec_i "SDSS_offsetDec_i",offsetDec_z "SDSS_offsetDec_z",primTarget "SDSS_primTarget",secTarget "SDSS_secTarget",extinction_u "SDSS_extinction_u",extinction_g "SDSS_extinction_g",extinction_r "SDSS_extinction_r",extinction_i "SDSS_extinction_i",extinction_z "SDSS_extinction_z",priority "SDSS_priority",rho "SDSS_rho",nProf_u "SDSS_nProf_u",nProf_g "SDSS_nProf_g",nProf_r "SDSS_nProf_r",nProf_i "SDSS_nProf_i",nProf_z "SDSS_nProf_z",loadVersion "SDSS_loadVersion",fieldID "SDSS_fieldID",parentID "SDSS_parentID",SpecObjID "SDSS_SpecObjID",u "SDSS_u",g "SDSS_g",r "SDSS_r",i "SDSS_i",z "SDSS_z",Err_u "SDSS_Err_u",Err_g "SDSS_Err_g",Err_r "SDSS_Err_r",Err_i "SDSS_Err_i",Err_z "SDSS_Err_z",dered_u "SDSS_dered_u",dered_g "SDSS_dered_g",dered_r "SDSS_dered_r",dered_i "SDSS_dered_i",dered_z "SDSS_dered_z" from PhotoObjAll where ((htmID >= 14468096458752 and htmID <= 14468100653055))'
]
