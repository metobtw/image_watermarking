import numpy as np
from PIL import Image


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
    def reverse_affine_func(red, green, blue, key):
        reversed_affine_matrix = np.array([[0, -1], [-1, -1]])
        renew_r, renew_b, renew_g = np.array(red), np.array(blue), np.array(green)
        for k in range(key):
            if k != 0:
                red, green, blue = np.copy(renew_r), np.copy(renew_g), np.copy(renew_b)
                renew_r, renew_g, renew_b = np.zeros([len(red), len(red)]), np.zeros(
                    [len(red), len(red)]), np.zeros([len(red), len(red)])
            for i in range(len(red)):
                for j in range(len(red)):
                    xy = np.array([[j], [i]])
                    if xy[0] + xy[1] <= len(red):
                        xy = reversed_affine_matrix.dot(xy) + np.array([[len(red)], [len(red)]])
                    else:
                        xy = reversed_affine_matrix.dot(xy) + np.array([[len(red)], [2 * len(red)]])
                    xy %= len(red)
                    renew_r[xy[1][0]][xy[0][0]] = red[i][j]
                    renew_g[xy[1][0]][xy[0][0]] = green[i][j]
                    renew_b[xy[1][0]][xy[0][0]] = blue[i][j]
        return renew_r, renew_g, renew_b


class PreprocessingWatermark:
    def __init__(self, route_to_img):
        self.r, self.g, self.b = AffineTransform(route_to_img).affine_func(1)

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
        size = (len(self.r) // 8) ** 0.5
        for i in range(0, len(self.r), 8):
            num_r, num_g, num_b = int(self.r[i:i + 8], 2), int(self.g[i:i + 8], 2), int(self.b[i:i + 8], 2)
            red_nums[index].append(num_r), green_nums[index].append(num_g), blue_nums[index].append(num_b)
            if len(red_nums[index]) == (len(self.r) // 8) ** 0.5:
                index += 1
                if index == (len(self.r) // 8) ** 0.5:
                    return red_nums, green_nums, blue_nums, int(size)
                red_nums.append([]), blue_nums.append([]), green_nums.append([])




class PreprocessingImage:
    # todo: rewrite obtaining_mec, extracting_watermark
    def __init__(self, route_to_image):
        self.r, self.g, self.b = ImageToRGB(route_to_image).converting_to_rgb()

    def obtaining_mec(self, route_to_watermark):
        quantization_step_blue = 10
        quantization_step_red, quantization_step_green = quantization_step_blue * 0.78, quantization_step_blue * 0.94
        quantization_coef = 0.25
        chosing_arr = generating_matrix_s(len(self.r))
        r_wat, g_wat, b_wat = PreprocessingWatermark(route_to_watermark).array_to_str()
        ind_layer = 0
        for i in range(len(chosing_arr)):
            if ind_layer == len(r_wat):
                break

            num_block = chosing_arr[i]
            block_w = num_block % (len(self.r) // 2)
            block_h = (num_block - block_w) // (len(self.r) // 2)

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
                    self.r[height][width] %= 255
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
                    self.g[height][width] %= 255

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
                    # print(self.b[height][width], self.b[height][width] + int(deltaE_blue))
                    self.b[height][width] += int(deltaE_blue)
                    self.b[height][width] %= 255
            # end-blue
            # print(int(deltaE_red),int(deltaE_green),int(deltaE_blue))
            ind_layer += 1
        # print(ind_layer)
        ImageToRGB.saving_image(self.r, self.g, self.b, len(self.r), 'saved_image.jpg')

    def extracting_watermark(self, route_to_s_matrix, step, watermark_size):
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
            # red
            Emax_red = (self.r[block_h * 2][block_w * 2] + self.r[block_h * 2][block_w * 2 + 1] +
                        self.r[block_h * 2 + 1][
                            block_w * 2] + self.r[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if Emax_red % step < 0.5 * step:
                bits_red += '0'
            else:
                bits_red += '1'
            # green
            Emax_green = (self.g[block_h * 2][block_w * 2] + self.g[block_h * 2][block_w * 2 + 1] +
                          self.g[block_h * 2 + 1][
                              block_w * 2] + self.g[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if Emax_green % step < 0.5 * step:
                bits_green += '0'
            else:
                bits_green += '1'
            # blue
            Emax_blue = (self.b[block_h * 2][block_w * 2] + self.b[block_h * 2][block_w * 2 + 1] +
                         self.b[block_h * 2 + 1][
                             block_w * 2] + self.b[block_h * 2 + 1][block_w * 2 + 1]) / 4
            if Emax_blue % step < 0.5 * step:
                bits_blue += '0'
            else:
                bits_blue += '1'
            bits_cnter += 1
        red_p, green_p, blue_p, size = ExtractingWatermark(bits_red, bits_green, bits_blue).str_to_array()
        print(len(blue_p))
        red_pix, green_pix, blue_pix = AffineTransform.reverse_affine_func(red_p, green_p, blue_p, 1)

        ImageToRGB.saving_image(red_pix, green_pix, blue_pix, size, 'saved_watermark.jpg')
        return bits_red, bits_green, bits_blue


def main():
    """
    todo asking route to image
    """

    # print(ImageToRGB('image.jpg').converting_to_rgb())
    # print(PreprocessingWatermark('image.jpg').array_to_str())
    # t = generating_matrix_s(512)
    # print(t.shape)
    PreprocessingImage('4.2.05.tiff').obtaining_mec('watermark.jpg')
    PreprocessingImage('saved_image.jpg').extracting_watermark('s_matrix.txt', 10, 32)


if __name__ == '__main__':
    main()
