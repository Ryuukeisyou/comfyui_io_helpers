import os
import json
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
from torch import Tensor
from comfy.cli_args import args
import base64
from io import BytesIO

class ImageLoadFromBase64:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_string": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    # RETURN_NAMES = ("any")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "io_helpers"

    def main(self, base64_string: str):
        # Remove the base64 prefix (e.g., "data:image/png;base64,")
        if (base64_string.startswith("data:image/")):
            _, base64_string = base64_string.split(",", 1)
        decoded_bytes = base64.b64decode(base64_string)
        file_like_object = BytesIO(decoded_bytes)
        try: 
            img = Image.open(file_like_object)
        except:
            return (None, None)

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

class ImageLoadByPath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    # RETURN_NAMES = ("any")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "io_helpers"

    def main(self, file_path: str):
        img = Image.open(file_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

class ImageLoadAsMaskByPath:
    _color_channels = ["alpha", "red", "green", "blue"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "channel": (cls._color_channels,)
            }
        }

    RETURN_TYPES = ("MASK",)
    # RETURN_NAMES = ("any")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "io_helpers"

    def main(self, file_path: str, channel):
        i = Image.open(file_path)
        i = ImageOps.exif_transpose(i)
        if i.getbands() != ("R", "G", "B", "A"):
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

class ImageSaveToPath:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "folder_path": ("STRING", {}),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI"
                }),
                "save_prompt": ("BOOLEAN", {
                    "default": True,
                }),
                "save_extra_pnginfo": ("BOOLEAN", {
                    "default": True,
                }),
                "compress_level": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 9,
                    "step": 1
                })
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"

    OUTPUT_NODE = True

    CATEGORY = "io_helpers"

    def main(
            self, 
            images: Tensor, 
            folder_path: str, 
            file_name: str, 
            prompt=None, 
            save_prompt=True, 
            extra_pnginfo=None, 
            save_extra_pnginfo=True, 
            compress_level=4):
        file_paths = []
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():
                    file_paths.append(entry.path)
        png_paths = filter(lambda x: x.endswith(".png"), file_paths)
        counter = 0
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if save_prompt and prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if save_extra_pnginfo and extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{file_name}_{counter:05}.png"
            full_file_path = os.path.join(folder_path, file)

            while(full_file_path in png_paths):
                counter += 1
                file = f"{file_name}_{counter:05}.png"
                full_file_path = os.path.join(folder_path, file)

            img.save(full_file_path, pnginfo=metadata, compress_level=compress_level)
            results.append({
                "filename": file,
                "folder": folder_path,
                "full_path": full_file_path,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class ImageSaveAsBase64:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "save_prompt": ("BOOLEAN", {
                    "default": True,
                }),
                "save_extra_pnginfo": ("BOOLEAN", {
                    "default": True,
                }),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"

    OUTPUT_NODE = True

    CATEGORY = "io_helpers"

    def main(
            self, 
            images: Tensor,
            save_prompt=True, 
            save_extra_pnginfo=True,
            prompt=None, 
            extra_pnginfo=None):
        
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if save_prompt and prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if save_extra_pnginfo and extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Create a BytesIO object to simulate a file-like object
            image_stream = BytesIO()

            # Save the image to the BytesIO stream
            img.save(image_stream, pnginfo=metadata, format="PNG")

            # Get raw bytes from the buffer
            image_bytes = image_stream.getvalue()

            # Encode the BytesIO stream content to base64
            base64_string = "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8") # Decode for text representation

            results.append({
                "base64_string": base64_string,
            })

        return { "ui": { "images": results } }
    
class VHSFileNamesToStrings:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vhs_filenames": ("VHS_FILENAMES", {}),
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"

    OUTPUT_NODE = True

    CATEGORY = "io_helpers"

    def main(
            self, 
            vhs_filenames:tuple):

        return { "ui": { "file_paths": vhs_filenames[1] } }
    
class TypeConversion:
    def __init__(self) -> None:
        
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["STRING", "INT", "FLOAT"], {}),
                "output_type": (["STRING", "INT", "FLOAT"], {})
            },
            "optional": {
                "string_value": ("STRING", {
                    "default": None,
                }),
                "int_value": ("INT", {
                    "default": None,
                }),
                "float_value": ("FLOAT", {
                    "default": None,
                    "step": 0.001
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "FLOAT")

    FUNCTION = "main"

    CATEGORY = "io_helpers"

    def main(self,
             input_type,
             output_type,
             string_value,
             int_value,
             float_value):
        input_value = None
        if (input_type) == "STRING":
            input_value = string_value
        if (input_type) == "INT":
            input_value = int_value
        if (input_type) == "FLOAT":
            input_value = float_value
        if (output_type) == "STRING":
            return (str(input_value), 0, 0)
        if (output_type) == "INT":
            return ("", int(input_value), 0)
        if (output_type) == "FLOAT":
            return ("", 0, float(input_value))
        

NODE_CLASS_MAPPINGS = {
    'ImageLoadFromBase64(IOHelpers)': ImageLoadFromBase64,
    'ImageLoadByPath(IOHelpers)': ImageLoadByPath,
    'ImageLoadAsMaskByPath(IOHelpers)': ImageLoadAsMaskByPath,
    'ImageSaveToPath(IOHelpers)': ImageSaveToPath,
    'ImageSaveAsBase64(IOHelpers)': ImageSaveAsBase64,
    'VHSFileNamesToStrings(IOHelpers)': VHSFileNamesToStrings,
    'TypeConversion(IOHelpers)': TypeConversion,
}