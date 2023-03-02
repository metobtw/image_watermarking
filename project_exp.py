import numpy as np
from PIL import Image
import math


def generating_matrix_s(size):
    """
    Image is square with even sides ( because block is 2x2 )
    generates array of block selection matrix ( randperm in MATLAB )

    :param size: height/weight of image
    :return: pseudo-randomly array of block selection
    """
    s_mat = np.random.permutation((size ** 2) // 4)
    f = open('s_matrix.txt', 'w')
    for i in range(len(s_mat)):
        f.write(str(s_mat[i]) + ' ')
    f.close()
    return s_mat


class ImageToRGB:
    """
    Opens an image, storing pixel values in arrays
    converting_to_rgb - saving values to arrays
    saving_image - from pixels values to RGB image
    """

    def __init__(self, route_to_pic):
        self.pic = Image.open(route_to_pic).convert('RGB')
        self.r, self.g, self.b = [], [], []

    def converting_to_rgb(self):
        for height in range(self.pic.height):
            self.r.append([]), self.g.append([]), self.b.append([])
            for width in range(self.pic.width):
                r_pixel, g_pixel, b_pixel = self.pic.getpixel((width, height))
                self.r[height].append(r_pixel), self.g[height].append(g_pixel), self.b[height].append(b_pixel)
        return self.r, self.g, self.b

    @staticmethod
    def saving_image(red, green, blue, size, output_name):
        to_out = np.zeros([size, size, 3], dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                to_out[i][j] = [red[i][j], green[i][j], blue[i][j]]
        img = Image.fromarray(to_out.astype(np.uint8))
        img.save(output_name)


class AffineTransform:
    # todo: affine transformation on limited grids https://ieeexplore.ieee.org/document/5945081
    def __init__(self, route_to_img):
        self.r, self.g, self.b = ImageToRGB(route_to_img).converting_to_rgb()

    def affine_func(self, key):
        affine_matrix = np.array([[1, -1], [-1, 0]])
        new_r, new_b, new_g = np.array(self.r), np.array(self.b), np.array(self.g)
        for k in range(key):
            if k != 0:
                self.r, self.g, self.b = np.copy(new_r), np.copy(new_g), np.copy(new_b)
                new_r, new_g, new_b = np.zeros([len(self.r), len(self.r)]), np.zeros(
                    [len(self.r), len(self.r)]), np.zeros([len(self.r), len(self.r)])
            for i in range(len(self.r)):
                for j in range(len(self.r)):
                    xy = np.array([[j], [i]])
                    if i < j:
                        xy = affine_matrix.dot(xy) + np.array([[len(self.r)], [len(self.r)]])
                    else:
                        xy = affine_matrix.dot(xy) + np.array([[0], [len(self.r)]])
                    xy %= len(self.r)
                    new_r[xy[1][0]][xy[0][0]] = self.r[i][j]
                    new_g[xy[1][0]][xy[0][0]] = self.g[i][j]
                    new_b[xy[1][0]][xy[0][0]] = self.b[i][j]
        return new_r, new_g, new_b

    @staticmethod
    def reverse_affine_func(r, g, b, key):
        reversed_affine_matrix = np.array([[0, -1], [-1, -1]])
        renew_r, renew_b, renew_g = np.array(r), np.array(b), np.array(g)
        for k in range(key):
            if k != 0:
                r, g, b = np.copy(renew_r), np.copy(renew_g), np.copy(renew_b)
                renew_r, renew_g, renew_b = np.zeros([len(r), len(r)]), np.zeros(
                    [len(r), len(r)]), np.zeros([len(r), len(r)])
            for i in range(len(r)):
                for j in range(len(r)):
                    xy = np.array([[j], [i]])
                    if xy[0] + xy[1] <= len(r):
                        xy = reversed_affine_matrix.dot(xy) + np.array([[len(r)], [len(r)]])
                    else:
                        xy = reversed_affine_matrix.dot(xy) + np.array([[len(r)], [2 * len(r)]])
                    xy %= len(r)
                    renew_r[xy[1][0]][xy[0][0]] = r[i][j]
                    renew_g[xy[1][0]][xy[0][0]] = g[i][j]
                    renew_b[xy[1][0]][xy[0][0]] = b[i][j]
        return renew_r, renew_g, renew_b


class PreprocessingWatermark:
    def __init__(self, route_to_img, key=0):
        # self.r, self.g, self.b = AffineTransform(route_to_img).affine_func(1)
        if key == 1:
            self.r, self.g, self.b = AffineTransform(route_to_img).affine_func(1)
        else:
            self.r, self.g, self.b = ImageToRGB(route_to_img).converting_to_rgb()

    def array_to_str(self):
        red, green, blue = '', '', ''
        for i in range(len(self.r)):
            for j in range(len(self.r)):
                red += f'{self.r[i][j]:08b}'
                green += f'{self.g[i][j]:08b}'
                blue += f'{self.b[i][j]:08b}'
        return red, green, blue


class ExtractingWatermark:
    def __init__(self, red, green, blue):
        self.r, self.g, self.b = red, green, blue

    def str_to_array(self):
        red_nums, green_nums, blue_nums = [[]], [[]], [[]]
        index = 0
        #print(len(self.r))
        size = int((len(self.r) // 8) ** 0.5)
        #print(size)
        for i in range(0, len(self.r), 8):
            num_r, num_g, num_b = int(self.r[i:i + 8], 2), int(self.g[i:i + 8], 2), int(self.b[i:i + 8], 2)
            red_nums[index].append(num_r), green_nums[index].append(num_g), blue_nums[index].append(num_b)
            #print(len(red_nums[index]), len(self.r))
            if len(red_nums[index]) == size:
                index += 1
                if index == size:
                    return red_nums, green_nums, blue_nums, int(size)
                red_nums.append([]), blue_nums.append([]), green_nums.append([])


class PreprocessingImage:
    def __init__(self, route_to_image):
        self.r, self.g, self.b = ImageToRGB(route_to_image).converting_to_rgb()

    def obtaining_mec(self, route_to_watermark, index_img, index_wat):
        quantization_step_blue = 24
        quantization_step_red, quantization_step_green = quantization_step_blue * 0.78, quantization_step_blue * 0.94
        quantization_coef = 0.25
        chosing_arr = generating_matrix_s(len(self.r))
        r_wat, g_wat, b_wat = PreprocessingWatermark(route_to_watermark, 1).array_to_str()
        # print(r_wat)
        ind_layer = 0
        for i in range(len(chosing_arr)):
            if ind_layer == len(r_wat):
                break

            num_block = chosing_arr[i]
            block_w = num_block % (len(self.r) // 2)
            block_h = (num_block - block_w) // (len(self.r) // 2)
            # print(block_w, block_h, end=' ', sep='boba')
            # red
            Emax_red = (self.r[block_h * 2][block_w * 2] + self.r[block_h * 2][block_w * 2 + 1] +
                        self.r[block_h * 2 + 1][
                            block_w * 2] + self.r[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if r_wat[ind_layer] == '0':
                Eupper_red = Emax_red - Emax_red % quantization_step_red + (
                        quantization_coef + 1) * quantization_step_red
                Elower_red = Emax_red - Emax_red % quantization_step_red + quantization_coef * quantization_step_red
            else:
                Eupper_red = Emax_red - Emax_red % quantization_step_red + (
                        1 - quantization_coef) * quantization_step_red
                Elower_red = Emax_red - Emax_red % quantization_step_red - quantization_coef * quantization_step_red
            Emax_red_dot = 0
            if abs(Emax_red - Eupper_red) <= abs(Emax_red - Elower_red):
                Emax_red_dot = Eupper_red
            else:
                Emax_red_dot = Elower_red
            deltaE_red = Emax_red_dot - Emax_red
            for height in range(block_h * 2, block_h * 2 + 2):
                for width in range(block_w * 2, block_w * 2 + 2):
                    self.r[height][width] += int(deltaE_red)
                    #self.r[height][width] %= 255
                    self.r[height][width] = abs(self.r[height][width]) % 255
            # end-red

            # green
            Emax_green = (self.g[block_h * 2][block_w * 2] + self.g[block_h * 2][block_w * 2 + 1] +
                          self.g[block_h * 2 + 1][
                              block_w * 2] + self.g[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if g_wat[ind_layer] == '0':
                Eupper_green = Emax_green - Emax_green % quantization_step_green + (
                        quantization_coef + 1) * quantization_step_green
                Elower_green = Emax_green - Emax_green % quantization_step_green + quantization_coef * quantization_step_green
            else:
                Eupper_green = Emax_green - Emax_green % quantization_step_green + (
                        1 - quantization_coef) * quantization_step_green
                Elower_green = Emax_green - Emax_green % quantization_step_green - quantization_coef * quantization_step_green
            Emax_green_dot = 0
            if abs(Emax_green - Eupper_green) <= abs(Emax_green - Elower_green):
                Emax_green_dot = Eupper_green
            else:
                Emax_green_dot = Elower_green
            deltaE_green = Emax_green_dot - Emax_green
            for height in range(block_h * 2, block_h * 2 + 2):
                for width in range(block_w * 2, block_w * 2 + 2):
                    self.g[height][width] += int(deltaE_green)
                    #self.g[height][width] %= 255
                    self.g[height][width] = abs(self.g[height][width]) % 255
            # end-green

            # blue
            Emax_blue = (self.b[block_h * 2][block_w * 2] + self.b[block_h * 2][block_w * 2 + 1] +
                         self.b[block_h * 2 + 1][
                             block_w * 2] + self.b[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if b_wat[ind_layer] == '0':
                Eupper_blue = Emax_blue - Emax_blue % quantization_step_blue + (
                        quantization_coef + 1) * quantization_step_blue
                Elower_blue = Emax_blue - Emax_blue % quantization_step_blue + quantization_coef * quantization_step_blue
            else:
                Eupper_blue = Emax_blue - Emax_blue % quantization_step_blue + (
                        1 - quantization_coef) * quantization_step_blue
                Elower_blue = Emax_blue - Emax_blue % quantization_step_blue - quantization_coef * quantization_step_blue
            Emax_blue_dot = 0
            if abs(Emax_blue - Eupper_blue) <= abs(Emax_blue - Elower_blue):
                Emax_blue_dot = Eupper_blue
            else:
                Emax_blue_dot = Elower_blue
            deltaE_blue = Emax_blue_dot - Emax_blue
            for height in range(block_h * 2, block_h * 2 + 2):
                for width in range(block_w * 2, block_w * 2 + 2):
                    #print(self.b[height][width], deltaE_blue)
                    self.b[height][width] += int(deltaE_blue)
                    #self.b[height][width] %= 255
                    self.b[height][width] = abs(self.b[height][width]) % 255
            # end-blue
            ind_layer += 1

        #    print(self.r[block_h * 2][block_w * 2], self.r[block_h * 2][block_w * 2 + 1],
        #         self.r[block_h * 2 + 1][
        #            block_w * 2], self.r[block_h * 2 + 1][block_w * 2 + 1], sep='-', end=' ')
        # print('-' * 1000)
        #print(len(r_wat))
        ImageToRGB.saving_image(self.r, self.g, self.b, len(self.r),
                                f'saves/saved_image{index_img + 1}_{index_wat + 1}.tiff')

    def extracting_watermark(self, route_to_s_matrix, step, watermark_size, index_img, index_wat):
        quantization_step_blue = step
        quantization_step_red, quantization_step_green = quantization_step_blue * 0.78, quantization_step_blue * 0.94
        file = open(route_to_s_matrix, 'r')
        s_matrix = [line.split() for line in file]
        file.close()
        bits_cnter = 0
        bits_red, bits_blue, bits_green = '', '', ''
        for i in range(len(s_matrix[0])):
            if bits_cnter == (watermark_size ** 2) * 8:
                break
            block_w = int(s_matrix[0][i]) % (len(self.r) // 2)
            block_h = (int(s_matrix[0][i]) - block_w) // (len(self.r) // 2)
            # print(block_w,block_h,end=' ',sep='boba')
            # red
            Emax_red = (self.r[block_h * 2][block_w * 2] + self.r[block_h * 2][block_w * 2 + 1] +
                        self.r[block_h * 2 + 1][
                            block_w * 2] + self.r[block_h * 2 + 1][block_w * 2 + 1]) / 4

            # print(self.r[block_h * 2][block_w * 2], self.r[block_h * 2][block_w * 2 + 1],
            #      self.r[block_h * 2 + 1][
            #         block_w * 2], self.r[block_h * 2 + 1][block_w * 2 + 1], sep='-', end=' ')
            if (Emax_red % quantization_step_red) < (0.5 * quantization_step_red):
                bits_red += '0'
            else:
                bits_red += '1'
            # green
            Emax_green = (self.g[block_h * 2][block_w * 2] + self.g[block_h * 2][block_w * 2 + 1] +
                          self.g[block_h * 2 + 1][
                              block_w * 2] + self.g[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if (Emax_green % quantization_step_green) < (0.5 * quantization_step_green):
                bits_green += '0'
            else:
                bits_green += '1'
            # blue
            Emax_blue = (self.b[block_h * 2][block_w * 2] + self.b[block_h * 2][block_w * 2 + 1] +
                         self.b[block_h * 2 + 1][
                             block_w * 2] + self.b[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if (Emax_blue % quantization_step_blue) < (0.5 * quantization_step_blue):
                bits_blue += '0'
            else:
                bits_blue += '1'
            bits_cnter += 1
        #
        # print(bits_red)
        #
        #print(len(bits_red),len(bits_green),len(bits_blue))
        red_pix, green_pix, blue_pix, size = ExtractingWatermark(bits_red, bits_green, bits_blue).str_to_array()
        red_pix, green_pix, blue_pix = AffineTransform.reverse_affine_func(red_pix, green_pix, blue_pix, 1)
        ImageToRGB.saving_image(red_pix, green_pix, blue_pix, size,
                                f'saves/saved_watermark{index_img + 1}_{index_wat + 1}.tiff')
        return bits_red, bits_green, bits_blue


class Metrics:
    def __init__(self, first_img, second_img):
        self.r1, self.g1, self.b1 = ImageToRGB(first_img).converting_to_rgb()
        self.r2, self.g2, self.b2 = ImageToRGB(second_img).converting_to_rgb()

    def psnr_metric(self):
        sum_elem = [0, 0, 0]
        for i in range(len(self.r1)):
            for j in range(len(self.r1)):
                sum_elem[0] += (self.r1[i][j] - self.r2[i][j]) ** 2
                sum_elem[1] += (self.g1[i][j] - self.g2[i][j]) ** 2
                sum_elem[2] += (self.b1[i][j] - self.b2[i][j]) ** 2
        for i in range(3):
            sum_elem[i] = 10 * math.log10((len(self.r1) ** 2 * 255 ** 2) / sum_elem[i])
        return sum(sum_elem) / 3

    def ncc_metric(self):
        s_lower1, s_lower2 = 0, 0
        s_upper = 0
        for i in range(len(self.r1)):
            for j in range(len(self.r1)):
                s_lower1 += self.r1[i][j] ** 2 + self.g1[i][j] ** 2 + self.b1[i][j] ** 2
                s_lower2 += self.r2[i][j] ** 2 + self.g2[i][j] ** 2 + self.b2[i][j] ** 2
                s_upper += self.r1[i][j] * self.r2[i][j] + self.g1[i][j] * self.g2[i][j] + self.b1[i][j] * self.b2[i][j]
        # print(s_lower1,s_lower2,s_upper)
        return s_upper / (s_lower1 ** 0.5 * s_lower2 ** 0.5)

    def ssim_metric(self):
        mean1, mean2 = 0, 0
        for i in range(len(self.r1)):
            for j in range(len(self.r1)):
                mean1 += self.r1[i][j] + self.g1[i][j] + self.b1[i][j]
                mean2 += self.r2[i][j] + self.g2[i][j] + self.b2[i][j]
        mean1 /= ((len(self.r1) ** 2) * 3)
        mean2 /= ((len(self.r1) ** 2) * 3)
        sd1, sd2 = 0, 0
        cov = 0
        for i in range(len(self.r1)):
            for j in range(len(self.r1)):
                sd1 += (((self.r1[i][j] + self.g1[i][j] + self.b1[i][j]) / 3 - mean1) ** 2)
                sd2 += (((self.r2[i][j] + self.g2[i][j] + self.b2[i][j]) / 3 - mean2) ** 2)
                cov += ((self.r1[i][j] + self.g1[i][j] + self.b1[i][j]) / 3 - mean1) * (
                        (self.r2[i][j] + self.g2[i][j] + self.b2[i][j]) / 3 - mean2)
        # print(disp1,disp2,cov)
        cov /= (len(self.r1) ** 2)
        sd1 = (sd1 / (len(self.r1) ** 2)) ** 0.5
        sd2 = (sd2 / (len(self.r1) ** 2)) ** 0.5
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        # print(mean1,mean2,cov,disp1,disp2,c1,c2)
        return ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / (
                (mean1 ** 2 + mean2 ** 2 + c1) * (sd1 ** 2 + sd2 ** 2 + c2))

    @staticmethod
    def ber_metric(first_img, second_img):
        r1, g1, b1 = PreprocessingWatermark(first_img).array_to_str()
        r2, g2, b2 = PreprocessingWatermark(second_img).array_to_str()
        cnt = 0
        for i in range(len(r1)):
            if r1[i] != r2[i]:
                cnt += 1
            if g1[i] != g2[i]:
                cnt += 1
            if b1[i] != b2[i]:
                cnt += 1
        return cnt / (len(r1) * 3)

    def capacity(self):
        return (len(self.r1) ** 2 * 3 * 8) / (len(self.r2) ** 2)


def main():
    """
    todo asking route to image
    """
    for image in range(11):
        for watermark in range(2,3):
            print(image+1,'index')
            PreprocessingImage(f'images/{image + 1}.tiff').obtaining_mec(f'watermarks/{watermark + 1}.tiff', image,
                                                                         watermark)
            PreprocessingImage(f'saves/saved_image{image + 1}_{watermark + 1}.tiff').extracting_watermark(
                's_matrix.txt', 24, 32, image, watermark)
            print(Metrics(f'images/{image + 1}.tiff', f'saves/saved_image{image + 1}_{watermark + 1}.tiff').psnr_metric(), 'PSNR-image')
            print(Metrics(f'images/{image + 1}.tiff', f'saves/saved_image{image + 1}_{watermark + 1}.tiff').ssim_metric(), 'SSIM-image')
            print(Metrics(f'images/{image + 1}.tiff', f'saves/saved_image{image + 1}_{watermark + 1}.tiff').ncc_metric(), 'ncc-image')
            print(Metrics.ber_metric(f'watermarks/{watermark + 1}.tiff', f'saves/saved_watermark{image + 1}_{watermark + 1}.tiff'), 'ber-water')

    # PreprocessingImage('4.2.05.tiff').obtaining_mec('newwater.tiff')
    # PreprocessingImage('saved_image.tiff').extracting_watermark('s_matrix.txt', 24, 32)
    #
    # print(Metrics('4.2.05.tiff', 'saved_image.tiff').psnr_metric(), 'PSNR-image')
    # print(Metrics('4.2.05.tiff', 'saved_image.tiff').ncc_metric(), 'ncc-image')
    # print(Metrics('newwater.tiff', 'saved_watermark.tiff').ncc_metric(), 'ncc-water')
    # # print(Metrics('newwater.tiff', 'saved_watermark.tiff').psnr_metric())
    # print(Metrics('4.2.05.tiff', 'saved_image.tiff').ssim_metric(), 'SSIM-image')
    # print(Metrics.ber_metric('newwater.tiff', 'saved_watermark.tiff'), 'ber-water')
    # print(Metrics('newwater.tiff', '4.2.05.tiff').capacity(), 'capacity')
def abc():
    PreprocessingImage(f'saves/saved_image10_2.jpg').extracting_watermark(
        's_matrix.txt', 24, 90, 10, 3)
    print(Metrics(f'watermarks/2.tiff', f'saves/saved_watermark11_4.tiff').ncc_metric(),
          'ncc-water')

if __name__ == '__main__':
    main()
   # abc()
