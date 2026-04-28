import os
import sys
import json
import glob
import argparse
import logging
import traceback
import numpy as np
import rawpy
from PIL import Image

try:
    import colour
except ImportError:
    print("Please install colour-science: pip install colour-science")
    sys.exit(1)

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    print("Please install PyTorch and torchvision.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper functions for CCT/Tint ---

def compute_gains(img_neutral, img_target):
    """
    Compute RGB gains between a neutrally rendered image and a target (WB corrected) image.
    Both images should be numpy arrays of shape (H, W, 3) and range [0, 1].
    """
    # Create mask to avoid clipped or very dark pixels
    mask = (img_neutral > 0.05) & (img_neutral < 0.95) & (img_target > 0.05) & (img_target < 0.95)
    
    # If mask is too small, fallback to a more relaxed mask
    if np.sum(mask) < 1000:
        mask = (img_neutral > 0.01) & (img_neutral < 0.99) & (img_target > 0.01) & (img_target < 0.99)

    if np.sum(mask) == 0:
        return 1.0, 1.0, 1.0
        
    R_mean_n = np.mean(img_neutral[:, :, 0][mask[:, :, 0]])
    G_mean_n = np.mean(img_neutral[:, :, 1][mask[:, :, 1]])
    B_mean_n = np.mean(img_neutral[:, :, 2][mask[:, :, 2]])

    R_mean_t = np.mean(img_target[:, :, 0][mask[:, :, 0]])
    G_mean_t = np.mean(img_target[:, :, 1][mask[:, :, 1]])
    B_mean_t = np.mean(img_target[:, :, 2][mask[:, :, 2]])

    R_gain = R_mean_t / (R_mean_n + 1e-8)
    G_gain = G_mean_t / (G_mean_n + 1e-8)
    B_gain = B_mean_t / (B_mean_n + 1e-8)

    # Normalize to G=1
    R_gain /= (G_gain + 1e-8)
    B_gain /= (G_gain + 1e-8)
    G_gain = 1.0

    return R_gain, G_gain, B_gain

def gains_to_cct_tint(R_gain, G_gain, B_gain):
    """
    Convert RGB gains (in sRGB space) to CCT (Kelvin) and Tint.
    """
    # Illuminant color in sRGB space
    R_ill = 1.0 / (R_gain + 1e-8)
    G_ill = 1.0 / (G_gain + 1e-8)
    B_ill = 1.0 / (B_gain + 1e-8)

    # Normalize illuminant so max is 1 to avoid out of bounds when converting
    max_val = max(R_ill, G_ill, B_ill)
    RGB = np.array([R_ill, G_ill, B_ill]) / max_val

    # sRGB to XYZ
    XYZ = colour.RGB_to_XYZ(
        RGB,
        colour.models.sRGB_COLOURSPACE.whitepoint,
        colour.models.sRGB_COLOURSPACE.whitepoint,
        colour.models.sRGB_COLOURSPACE.matrix_RGB_to_XYZ
    )

    # XYZ to xy
    xy = colour.XYZ_to_xy(XYZ)

    # xy to CCT and Duv
    try:
        CCT, Duv = colour.xy_to_CCT(xy, 'Ohno 2013')
    except Exception:
        # Fallback if Ohno 2013 fails to converge
        try:
            CCT, Duv = colour.xy_to_CCT(xy, 'Robertson 1968')
        except Exception:
            return 5500, 0

    # Convert Duv to Lightroom/ACR Tint scale (-150 to 150)
    # Positive Duv is green, negative is magenta.
    # In Lightroom, positive Tint is magenta, negative is green.
    tint = -Duv * 3000

    return float(CCT), float(tint)

# --- Deep WB Setup ---

# We dynamically import the architecture to avoid requiring it in the same dir 
# if packaged differently, but assume it's in the 'arch' folder next to the script.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arch'))
try:
    from deep_wb_single_task import deepWBnet
except ImportError:
    deepWBnet = None
    logging.warning("deepWBnet not found. DWB method will not work. Ensure 'arch' folder is present.")

def load_dwb_model(model_path, device):
    if deepWBnet is None:
        raise RuntimeError("Deep WB architecture not found.")
    
    net = deepWBnet()
    if device == 'cuda' and torch.cuda.is_available():
        net = net.cuda()
        checkpoint = torch.load(model_path)
    else:
        net = net.cpu()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    return net

def process_dwb(img_neutral_np, net, device):
    """
    Run the neutral image through the Deep WB model.
    """
    img_tensor = transforms.ToTensor()(img_neutral_np).unsqueeze(0)
    if device == 'cuda' and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    with torch.no_grad():
        out = net(img_tensor, task=0) # task=0 is AWB
        out_awb = out[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    out_awb = np.clip(out_awb, 0, 1)
    return out_awb

# --- XMP Generation ---

def write_xmp(raw_path, temp, tint):
    xmp_path = os.path.splitext(raw_path)[0] + ".xmp"
    
    # Standard Adobe XMP template for White Balance
    xmp_content = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:08:21        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
   <crs:WhiteBalance>Custom</crs:WhiteBalance>
   <crs:Temperature>{int(round(temp))}</crs:Temperature>
   <crs:Tint>{int(round(tint))}</crs:Tint>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    with open(xmp_path, "w", encoding="utf-8") as f:
        f.write(xmp_content)
    logging.info(f"Wrote XMP: {xmp_path} (Temp: {int(round(temp))}K, Tint: {int(round(tint))})")

# --- Main Logic ---

def process_file(raw_path, method, config, net=None):
    logging.info(f"Processing: {raw_path} using method {method}")
    try:
        with rawpy.imread(raw_path) as raw:
            # Render neutral image
            neutral_rgb = raw.postprocess(
                user_wb=[1.0, 1.0, 1.0, 1.0], 
                no_auto_bright=True,
                output_bps=8,
                output_color=rawpy.ColorSpace.sRGB
            ).astype(np.float32) / 255.0

            if method == 'ASH':
                # Use As-Shot (Camera WB)
                target_rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    output_color=rawpy.ColorSpace.sRGB
                ).astype(np.float32) / 255.0
            
            elif method == 'AWB':
                # Use rawpy's Auto WB
                target_rgb = raw.postprocess(
                    use_auto_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    output_color=rawpy.ColorSpace.sRGB
                ).astype(np.float32) / 255.0
            
            elif method == 'DWB':
                # Use Deep White-Balance Model
                if net is None:
                    raise ValueError("Deep WB model not loaded.")
                target_rgb = process_dwb(neutral_rgb, net, config.get("device", "cpu"))
            
            else:
                raise ValueError(f"Unknown method: {method}")

            # Compute gains between neutral and target
            r_g, g_g, b_g = compute_gains(neutral_rgb, target_rgb)
            
            # Convert gains to CCT and Tint
            cct, tint = gains_to_cct_tint(r_g, g_g, b_g)

            # Apply presets
            preset_name = config.get("preset", "true_neutral")
            intensity = config.get("intensity", 0) / 100.0
            presets = config.get("presets", {})
            
            if preset_name in presets:
                preset_data = presets[preset_name]
                # If preset modifies temperature by percentage
                shift_pct = preset_data.get("temp_shift_percent", 0.0)
                # Scale shift by intensity (e.g., if shift is 15%, and intensity is 50%, actual shift is 7.5%)
                # Wait, intensity in config is maybe absolute, let's treat intensity as a multiplier.
                # If intensity is just a level (5, 10, 15, 20), we map it directly or use config.
                cct = cct * (1.0 + shift_pct)
                
                tint_shift = preset_data.get("tint_shift", 0.0)
                tint = tint + tint_shift

            # Clamp values
            clamp_min = config.get("clamp_min", 2000)
            clamp_max = config.get("clamp_max", 10000)
            cct = max(clamp_min, min(clamp_max, cct))
            tint = max(-150, min(150, tint))

            # Write XMP
            write_xmp(raw_path, cct, tint)

    except Exception as e:
        logging.error(f"Failed to process {raw_path}: {str(e)}")
        traceback.print_exc()
        # Fallback XMP on failure
        logging.warning(f"Creating fallback XMP for {raw_path} (5500K, 0 Tint)")
        write_xmp(raw_path, 5500, 0)

def main():
    parser = argparse.ArgumentParser(description="AI White Balance Tool for RAW images.")
    parser.add_argument("input", help="Input raw file or directory")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--method", choices=["ASH", "AWB", "DWB"], default="DWB", help="White balance method to use")
    
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        logging.warning(f"Config file {args.config} not found. Using defaults.")

    # Prepare model if DWB is selected
    net = None
    if args.method == "DWB":
        model_path = config.get("model_path", "models/net_awb.pth")
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}. Please download it and update config.json.")
            if args.method == "DWB":
                logging.error("Cannot use DWB method without a model.")
                sys.exit(1)
        else:
            logging.info(f"Loading model from {model_path}...")
            net = load_dwb_model(model_path, config.get("device", "cpu"))

    # Collect files
    raw_exts = (".cr2", ".nef")
    files_to_process = []
    
    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(raw_exts):
                    files_to_process.append(os.path.join(root, file))
    elif os.path.isfile(args.input):
        if args.input.lower().endswith(raw_exts):
            files_to_process.append(args.input)
        else:
            logging.error("Input file must be a .CR2 or .NEF file.")
            sys.exit(1)
    else:
        logging.error("Input path does not exist.")
        sys.exit(1)

    if not files_to_process:
        logging.warning("No RAW files found to process.")
        sys.exit(0)

    for idx, f in enumerate(files_to_process):
        logging.info(f"[{idx+1}/{len(files_to_process)}]")
        process_file(f, args.method, config, net=net)

    logging.info("Done.")

if __name__ == "__main__":
    main()
