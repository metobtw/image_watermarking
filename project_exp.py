from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt


def saving(arr, name):
    """Takes an array of pixels, converts to an image and saves with a name 'name'

    Args:
        arr : Array of pixel values
        name: How we want to name the picture
    """
    Image.fromarray(arr).save(name)


class Watermark:
    def __init__(self, route: str):
        """Reading image as a pixel value

        Args:
            route (str): image path
        """
        self.watermark = np.asarray(Image.open(route))

    def watermark_string(self):
        """We take an array of pixels, convert each layer (red, blue, green) into a binary string.

        Returns:
            string_wat: Two dimensional array with binary strings
        """
        string_wat = ['', '', '']
        for i in range(len(self.watermark)):
            for j in range(len(self.watermark)):
                for layer in range(3):
                    string_wat[layer] += f'{self.watermark[i][j][layer]:08b}'
        return string_wat


class ObtainingWatermark:
    def __init__(self, route_wat: str, route_img: str, quantization_step_blue: int):
        """Open the watermark and convert it into an array of binary strings, the image into an array of pixels.
        We create an array of permutations 2*2 blocks and save it to a file.

        Args:
            route_wat (str): watermark path
            route_img (str): image path
            quantization_step_blue (int): blue layer quantization step
        """
        self.b_st = quantization_step_blue
        self.wat = Watermark(route_wat).watermark_string()
        img = Image.open(route_img).convert('RGB')
        self.img = np.array(img)
        self.s_mat = np.random.permutation((len(self.img) ** 2) // 4)
        with open("s-matrix.txt", "w") as txt_file:
            for line in self.s_mat:
                txt_file.write("".join(str(line)) + ' ')

    def obtaining(self, route : str, alpha=0.25):
        """We go through the blocks of the permutation matrix until we run out of bits to embed in the watermark array. 
        We calculate the maximum coefficient for each block and add a delta for each pixel value in the block. 
        Then we save the image.

        Args:
            route (str): image path
            alpha (float, optional): Alpha coef for the quantization step. Defaults to 0.25.
        """
        quant_steps = [self.b_st * 0.78, self.b_st * 0.94, self.b_st]
        ind = 0
        for block in self.s_mat:
            if ind == len(self.wat[0]):
                break
            block_w = block % (len(self.img) // 2)
            block_h = (block - block_w) // (len(self.img) // 2)
            for layer in range(3):
                E_max = 0
                for i in range(block_h * 2, block_h * 2 + 2):
                    for j in range(block_w * 2, block_w * 2 + 2):
                        E_max += self.img[i][j][layer]
                E_max /= 4
                if self.wat[layer][ind] == '0':
                    E_upper = E_max - round((E_max % quant_steps[layer]), 2) + (alpha + 1) * quant_steps[layer]
                    E_lower = E_max - round((E_max % quant_steps[layer]), 2) + alpha * quant_steps[layer]
                else:
                    E_upper = E_max - round((E_max % quant_steps[layer]), 2) + (1 - alpha) * quant_steps[layer]
                    E_lower = E_max - round((E_max % quant_steps[layer]), 2) - alpha * quant_steps[layer]

                if abs(E_max - E_upper) <= abs(E_max - E_lower):
                    E_max_dot = E_upper
                else:
                    E_max_dot = E_lower

                delta_E = int(E_max_dot - E_max)
                for i in range(block_h * 2, block_h * 2 + 2):
                    for j in range(block_w * 2, block_w * 2 + 2):
                        if self.img[i][j][layer] + delta_E < 0 or self.img[i][j][layer] + delta_E > 255:
                            delta_E = -delta_E
                        self.img[i][j][layer] += delta_E
                        self.img[i][j][layer] %= 256
            ind += 1
        saving(self.img, route)


class ExtractingWatermark:
    def __init__(self, route_s_matrix: str, route_img: str, quantization_step_blue: int, size: int):
        """Opens and transforms the image into an array of pixels opens a permutation matrix.

        Args:
            route_s_matrix (str): route to block permutation matrix
            route_img (str): image path
            quantization_step_blue (int): blue channel quantization step
            size (int): the size of the watermark in the image
        """
        img = Image.open(route_img).convert('RGB')
        self.img = np.array(img)
        self.s_mat = np.loadtxt(route_s_matrix, dtype=int)
        self.b_st = quantization_step_blue
        self.arr_str = ['', '', '']
        self.size = size

    def extracting(self, route : str):
        """For each image block, we calculate the maximum coefficient (E_max). 
        We read the extracted bit based on a comparison of the maximum coefficient and the quantization step.
        Later, we create an array (wat) where the received decimal values of the watermark are written. Convert to image.

        Args:
            route (str): watermark path
        """
        quant_steps = [self.b_st * 0.78, self.b_st * 0.94, self.b_st]
        ind = 0
        for block in self.s_mat:
            if ind == (self.size ** 2) * 8:
                break
            block_w = block % (len(self.img) // 2)
            block_h = (block - block_w) // (len(self.img) // 2)
            for layer in range(3):
                E_max = 0
                for i in range(block_h * 2, block_h * 2 + 2):
                    for j in range(block_w * 2, block_w * 2 + 2):
                        E_max += self.img[i][j][layer]
                E_max /= 4
                if (E_max % quant_steps[layer]) < quant_steps[layer] / 2:
                    self.arr_str[layer] += '0'
                else:
                    self.arr_str[layer] += '1'

            ind += 1

        wat = np.zeros([self.size, self.size, 3], dtype=np.uint8)
        for layer in range(3):
            ind1, ind2 = 0, 0
            for i in range(0, len(self.arr_str[0]), 8):
                wat[ind1][ind2][layer] = int(self.arr_str[layer][i:i + 8], 2)
                if ind2 % (self.size - 1) == 0 and ind2 != 0:
                    ind2 = 0
                    ind1 += 1
                else:
                    ind2 += 1
        saving(wat, route)


def compress_img(image_name, quality, format, new_filename):
    """Compressing image to another file format.

    Args:
        image_name : name of the image
        quality : new image quality (0,100)
        format : file format.
        new_filename : name of new file
    """
    img = Image.open(image_name)
    img.save(new_filename, format=format, quality=quality, subsampling=0,quality_layers = (5,1))


class Metrics:
    def __init__(self, route1 : str, route2 : str):
        """Saving 2 images as an array of pixels.

        Args:
            route1 (str): image 1 path
            route2 (str): image 2 path
        """
        self.img1 = np.array(Image.open(route1).convert('RGB'), dtype=np.int64)
        self.img2 = np.array(Image.open(route2).convert('RGB'), dtype=np.int64)

    def psnr_metric(self):
        # Return PSNR Metric value

        sum_elem = [0, 0, 0]
        for i in range(len(self.img1)):
            for j in range(len(self.img1)):
                sum_elem[0] += (self.img1[i][j][0] - self.img2[i][j][0]) ** 2
                sum_elem[1] += (self.img1[i][j][1] - self.img2[i][j][1]) ** 2
                sum_elem[2] += (self.img1[i][j][2] - self.img2[i][j][2]) ** 2
        for i in range(3):
            sum_elem[i] = 10 * math.log10((len(self.img1) ** 2 * 255 ** 2) / sum_elem[i])
        return sum(sum_elem) / 3

    def ncc_metric(self):
        # Return NCC Metric value

        s_lower1, s_lower2 = 0, 0
        s_upper = 0
        for i in range(len(self.img1)):
            for j in range(len(self.img1)):
                s_lower1 += np.square(self.img1[i][j]).sum()
                s_lower2 += np.square(self.img2[i][j]).sum()
                s_upper = s_upper + self.img1[i][j][0] * self.img2[i][j][0] + self.img1[i][j][1] * self.img2[i][j][1] + \
                          self.img1[i][j][2] * self.img2[i][j][2]

        return s_upper / (s_lower1 ** 0.5 * s_lower2 ** 0.5)

    def ssim_metric(self):
        # Return SSIM Metric value

        mean1, mean2 = np.sum(self.img1), np.sum(self.img2)
        mean1 /= ((len(self.img1) ** 2) * 3)
        mean2 /= ((len(self.img1) ** 2) * 3)

        sd1, sd2 = 0, 0
        cov = 0

        for i in range(len(self.img1)):
            for j in range(len(self.img1)):
                sd1 += (sum(self.img1[i][j]) / 3 - mean1) ** 2
                sd2 += (sum(self.img2[i][j]) / 3 - mean2) ** 2
                cov += (sum(self.img1[i][j]) / 3 - mean1) * (sum(self.img2[i][j]) / 3 - mean2)
        cov /= (len(self.img1) ** 2)
        sd1 = (sd1 / (len(self.img1) ** 2)) ** 0.5
        sd2 = (sd2 / (len(self.img2) ** 2)) ** 0.5
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / (
                (mean1 ** 2 + mean2 ** 2 + c1) * (sd1 ** 2 + sd2 ** 2 + c2))

    @staticmethod
    def ber_metric(route1, route2):
        # Return BER Metric value

        w1 = Watermark(route1).watermark_string()
        w2 = Watermark(route2).watermark_string()
        cnt = 0
        for i in range(len(w1[0])):
            for layer in range(3):
                if w1[layer][i] != w2[layer][i]:
                    cnt += 1
        return cnt / (len(w1[0]) * 3)

    def pixel_count1(self):
        """For each layer we count the pixel values.

        Returns:
            np.array: pixel value arrays
        """
        d_red1, d_green1, d_blue1 = np.zeros(256, dtype=int), np.zeros(256, dtype=int), np.zeros(256, dtype=int)
        d_red2, d_green2, d_blue2 = np.zeros(256, dtype=int), np.zeros(256, dtype=int), np.zeros(256, dtype=int)
        for i in range(len(self.img1)):
            for j in range(len(self.img1)):
                temp1, temp2 = self.img1[i][j], self.img2[i][j]
                d_red1[temp1[0]] += 1
                d_red2[temp2[0]] += 1
                d_green1[temp1[1]] += 1
                d_green2[temp2[1]] += 1
                d_blue1[temp1[2]] += 1
                d_blue2[temp2[2]] += 1
        return d_red1, d_red2, d_green1, d_green2, d_blue1, d_blue2


class Experiments:
    @staticmethod
    def psnr_step():
        """Watermarks with sizes 90*90 and 32*32 are embedded in two images 512*512. 
        Moreover, each watermark is embedded with a quantization step ranging from 3 to 40.
        """
        for image in [4, 11]:
            for watermark in [1, 5]:
                arr = []
                for step in range(3, 40):
                    obt = ObtainingWatermark(f'watermarks/{watermark}.tiff', f'images/{image}.tiff', step)
                    obt.obtaining(route=f'experiment1/{image}.tiff')
                    ExtractingWatermark('s-matrix.txt', f'experiment1/{image}.tiff', step,
                                        int((len(obt.wat[0]) // 8) ** 0.5)).extracting(f'experiment1/{watermark}.tiff')
                    arr.append(Metrics(f'images/{image}.tiff', f'experiment1/{image}.tiff').psnr_metric())
                print(arr)
                plt.plot(range(3, 40), arr)
                plt.ylabel('PSNR')
                plt.xlabel('Quantization step')
                plt.grid(axis='both')
                plt.yticks(np.arange(int(min(arr)), int(max(arr)) + 1, 2.0))
                plt.xticks(np.arange(3, 41, 2.0))
                plt.title(f'Image number {image} + watermark number {watermark}')
                plt.savefig(f'experiment1/{image}_{watermark}.png')
                plt.clf()

    @staticmethod
    def metrics_wat_im():
        """Three watermarks with sizes of 32*32 are embedded in eleven 512*512 images, the quantization step is 24. 
        PSNR, SSIM, NCC, BER metrics are calculated. 
        A table is built based on the data.
        """
        for watermark in [3, 4, 5]:
            ssim, psnr, ber, ncc = [], [], [], []
            for image in range(1, 12):
                obt = ObtainingWatermark(f'watermarks/{watermark}.tiff', f'images/{image}.tiff', 24)
                obt.obtaining(route=f'experiment2/{image}.tiff')
                ExtractingWatermark('s-matrix.txt', f'experiment2/{image}.tiff', 24,
                                    int((len(obt.wat[0]) // 8) ** 0.5)).extracting(f'experiment2/w{watermark}.tiff')
                im_met = Metrics(f'images/{image}.tiff', f'experiment2/{image}.tiff')
                ssim.append(im_met.ssim_metric())
                psnr.append(im_met.psnr_metric())
                ncc.append(im_met.ncc_metric())
                ber.append(Metrics.ber_metric(f'watermarks/{watermark}.tiff', f'experiment2/w{watermark}.tiff'))
            print(ssim, psnr, ncc, ber, sep='\n')
            print('-' * 100)

    @staticmethod
    def pixel_values():
        """Various watermarks with different quantization steps are embedded in the image. 
        The dependence of pixel values before and after embedding is built.
        """
        import seaborn as sns
        for image in [11]:
            for watermark in [1, 5]:
                for step in [24, 40]:
                    obt = ObtainingWatermark(f'watermarks/{watermark}.tiff', f'images/{image}.tiff', step)
                    obt.obtaining(route=f'experiment3/{image}.tiff')
                    r1, r2, g1, g2, b1, b2 = Metrics(f'images/{image}.tiff', f'experiment3/{image}.tiff').pixel_count1()
                    # r1, r2, g1, g2, b1, b2 = sorted(r1.items()), sorted(r2.items()), sorted(g1.items()), sorted(
                    # g2.items()), sorted(b1.items()), sorted(b2.items())
                    sns.set(rc={"figure.figsize": (60, 20)})
                    sns.barplot(x=np.arange(0, 256), y=r1, color='g', alpha=0.6)
                    sns.barplot(x=np.arange(0, 256), y=r2, color='r', alpha=0.6)
                    plt.ylabel('pixel count', fontsize=20)
                    plt.xlabel('pixel value', fontsize=20)
                    plt.xticks(np.arange(0, 256), rotation=90)
                    plt.title(f'Image number {image} + watermark number {watermark} red', fontsize=30)
                    plt.savefig(f'experiment3/{image}_{watermark}_{step}_r.png')
                    plt.clf()

                    sns.barplot(x=np.arange(0, 256), y=g1, color='g', alpha=0.6)
                    sns.barplot(x=np.arange(0, 256), y=g2, color='r', alpha=0.6)
                    plt.ylabel('pixel count', fontsize=20)
                    plt.xlabel('pixel value', fontsize=20)
                    plt.xticks(np.arange(0, 256), rotation=90)
                    plt.title(f'Image number {image} + watermark number {watermark} green', fontsize=30)
                    plt.savefig(f'experiment3/{image}_{watermark}_{step}_g.png')
                    plt.clf()

                    sns.barplot(x=np.arange(0, 256), y=b1, color='g', alpha=0.6)
                    sns.barplot(x=np.arange(0, 256), y=b2, color='r', alpha=0.6)
                    plt.ylabel('pixel count', fontsize=20)
                    plt.xlabel('pixel value', fontsize=20)
                    plt.xticks(np.arange(0, 256), rotation=90)
                    plt.title(f'Image number {image} + watermark number {watermark} blue', fontsize=30)
                    plt.savefig(f'experiment3/{image}_{watermark}_{step}_b.png')
                    plt.clf()

    @staticmethod
    def after_jpeg():
        """A single watermarked image is compressed using the JPEG method and the NCC and BER metrics for the watermark are calculated.
        The image is compressed with quality parameters equal to 100,80,60,40,20.
        """
        ncc = []
        ber = []
        for quality in range(100, 19, -20):
            obt = ObtainingWatermark(f'watermarks/5.tiff', f'images/11.tiff', 24)
            obt.obtaining(route=f'experiment4/5.tiff')
            compress_img('experiment4/5.tiff', quality, 'JPEG', 'experiment4/5_compressed.jpeg')
            ExtractingWatermark('s-matrix.txt', 'experiment4/5_compressed.jpeg', 24,
                                int((len(obt.wat[0]) // 8) ** 0.5)).extracting(f'experiment4/5.tiff')
            im_met = Metrics(f'watermarks/5.tiff', f'experiment4/5.tiff')
            ncc.append(im_met.ncc_metric())
            ber.append(im_met.ber_metric(f'watermarks/5.tiff', f'experiment4/5.tiff'))
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(np.arange(100, 19, -20), ncc)
        ax1.set(ylabel='ncc', title='NCC/BER metric based on jpeg compression')
        ax2.plot(np.arange(100, 19, -20), ber)
        ax2.set(xlabel='quality', ylabel='ber')
        plt.savefig(f'experiment4/ncc-ber.png')
        plt.clf()

    @staticmethod
    def after_jpeg2000():
        """The watermarked image is compressed using the JPEG2000 method.
        Later, the watermark is extracted and NCC/BER metrics are calculated for it.
        """
        ber = []
        ncc = []
        for watermark in [2, 3]:
            for image in range(1, 12):
                obt = ObtainingWatermark(f'watermarks/{watermark}.tiff', f'images/{image}.tiff', 24)
                obt.obtaining(route=f'experiment5/{image}.tiff')
                compress_img(f'experiment5/{image}.tiff', 100, 'JPEG2000', f'experiment5/{image}_compressed.jp2')
                ExtractingWatermark('s-matrix.txt', f'experiment5/{image}_compressed.jp2', 24,
                                    int((len(obt.wat[0]) // 8) ** 0.5)).extracting(f'experiment5/w{watermark}.tiff')
                ber.append(Metrics.ber_metric(f'watermarks/{watermark}.tiff', f'experiment5/w{watermark}.tiff'))
                ncc.append(Metrics(f'watermarks/{watermark}.tiff', f'experiment5/w{watermark}.tiff').ncc_metric())
            ber.append('---')
            ncc.append('---')
        f = open('experiment5/out.txt', 'w')
        f.write(str(ber) + '\n' + str(ncc))
        f.close()


def main():
    Experiments.pixel_values()


if __name__ == '__main__':
    main()
