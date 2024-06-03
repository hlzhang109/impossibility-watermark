# This is here to fix circular imports.

from watermarkers import UMDWatermarker, UnigramWatermarker, EXPWatermarker, SemStampWatermarker

def get_watermarker(cfg, **kwargs):
    if cfg.watermark_args.name == "umd":
        return UMDWatermarker(cfg, **kwargs)
    elif cfg.watermark_args.name == "unigram":
        return UnigramWatermarker(cfg, **kwargs)
    elif cfg.watermark_args.name == "exp":
        return EXPWatermarker(cfg, **kwargs)
    elif cfg.watermark_args.name == "semstamp_lsh":
        return SemStampWatermarker(cfg, **kwargs)
    else:
        raise NotImplementedError