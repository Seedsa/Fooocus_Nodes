def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image"""
    from nodes import PreviewImage, SaveImage
    if output_type == "Hide":
        return list()
    if output_type == "Preview":
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
