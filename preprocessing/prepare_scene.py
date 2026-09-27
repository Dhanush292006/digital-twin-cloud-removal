"""Package a LISS-IV/other multispectral GeoTIFF target pair as a project NPZ."""
import argparse, json
from pathlib import Path
import numpy as np
from preprocessing.cloud_mask import estimate_mask


def read(path):
    try: import rasterio
    except ImportError as exc: raise RuntimeError("Install rasterio for GeoTIFF preparation") from exc
    with rasterio.open(path) as src: return src.read().astype(np.float32),src.profile.copy(),src.nodata


def align(arr,src_profile,dst_profile,resampling):
    import rasterio
    from rasterio.warp import reproject
    if (src_profile["crs"],src_profile["transform"],src_profile["width"],src_profile["height"]) == (dst_profile["crs"],dst_profile["transform"],dst_profile["width"],dst_profile["height"]): return arr
    out=np.zeros((arr.shape[0],dst_profile["height"],dst_profile["width"]),np.float32)
    for i in range(arr.shape[0]):
        reproject(arr[i],out[i],src_transform=src_profile["transform"],src_crs=src_profile["crs"],dst_transform=dst_profile["transform"],dst_crs=dst_profile["crs"],resampling=resampling)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cloudy",required=True); ap.add_argument("--clear",required=True); ap.add_argument("--output",required=True); ap.add_argument("--mask"); ap.add_argument("--scale",type=float,default=10000.0); ap.add_argument("--cloudy-key",default="cloudy"); ap.add_argument("--clear-key",default="clear"); a=ap.parse_args()
    import rasterio
    cloudy,profile,nodata=read(a.cloudy); clear,cp,clear_nodata=read(a.clear)
    if (nodata is not None and np.any(cloudy==nodata)) or (clear_nodata is not None and np.any(clear==clear_nodata)):
        raise ValueError("Source contains nodata pixels. Crop/reproject to a fully valid common footprint before packaging; do not treat nodata as zero reflectance.")
    if not np.isfinite(cloudy).all() or not np.isfinite(clear).all():
        raise ValueError("Source contains NaN/Inf. Repair or crop missing pixels before packaging.")
    cloudy=cloudy/a.scale; clear=align(clear,cp,profile,rasterio.enums.Resampling.bilinear)/a.scale
    if clear.shape!=cloudy.shape: raise ValueError("Cloudy/clear band counts must match; select a consistent set before packaging")
    if a.mask:
        mask,mp,_=read(a.mask); mask=align(mask,mp,profile,rasterio.enums.Resampling.nearest)[0].astype(np.uint8)
        if mask.max()>2: raise ValueError("Mask values must be 0 clear, 1 cloud, 2 shadow; remap source QA flags explicitly")
    else: mask=estimate_mask(cloudy)
    cloudy=np.clip(np.nan_to_num(cloudy),0,1).astype(np.float32); clear=np.clip(np.nan_to_num(clear),0,1).astype(np.float32)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**{a.cloudy_key:cloudy,a.clear_key:clear, "mask":mask})
    meta={"source_cloudy":str(Path(a.cloudy).resolve()),"source_clear":str(Path(a.clear).resolve()),"source_mask":str(Path(a.mask).resolve()) if a.mask else None,"crs":str(profile["crs"]),"transform":list(profile["transform"]),"nodata":nodata,"scale":a.scale,"shape":list(cloudy.shape),"mask_source":"provided" if a.mask else "heuristic; validate before scientific use"}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)); print(f"Wrote {out} and metadata sidecar")


if __name__=="__main__": main()
