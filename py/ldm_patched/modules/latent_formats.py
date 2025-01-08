
class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_dimensions = 2
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor

class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]
        self.taesd_decoder_name = "taesd_decoder"

class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3651,  0.4232,  0.4341],
                    [-0.2533, -0.0042,  0.1068],
                    [ 0.1076,  0.1111, -0.0362],
                    [-0.3165, -0.2492, -0.2188]
                ]
        self.latent_rgb_factors_bias = [ 0.1084, -0.0175, -0.0011]

        self.taesd_decoder_name = "taesdxl_decoder"

class SD_X4(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.08333
        self.latent_rgb_factors = [
            [-0.2340, -0.3863, -0.3257],
            [ 0.0994,  0.0885, -0.0908],
            [-0.2833, -0.2349, -0.3741],
            [ 0.2523, -0.0055, -0.1651]
        ]
