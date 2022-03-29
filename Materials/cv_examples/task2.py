import numpy as np
import cv2

from preprocessing import convolve, imshow


def sp_noise(image: np.ndarray, prob: float = 0.1, mode: str = "gray"):
    """
    Add salt and pepper noise to image
    :param image: input image
    :param prob: probability of the noise happening in one pixel
    :param mode: either 'gray' or 'colour' - controls way of applying sp noise
    :return: noisy image
    """
    output = image.copy()
    if mode == "gray":
        if len(image.shape) == 2:  # Gray
            black = 0
            white = 255
        else:
            if image.shape[2] == 3:  # RGB
                black = np.array([0, 0, 0], dtype=np.uint8)
                white = np.array([255, 255, 255], dtype=np.uint8)
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype=np.uint8)
                white = np.array([255, 255, 255, 255], dtype=np.uint8)
        probs = np.random.random(output.shape[:2])
    else:
        black = 0
        white = 255
        probs = np.random.random(output.shape)
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


def gauss_noise(image: np.ndarray, sigma: float = 50.):
    """
    Add Gaussian noise to image
    :param image: input image
    :param sigma: standard deviation for normal distribution
    :return: noisy image
    """
    output = image.copy()
    gauss = np.random.normal(0., sigma, output.shape)
    output = np.clip(output.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    return output


def deconvolution(image: np.ndarray, kernel: np.ndarray):
    """

    :param image:
    :param kernel:
    :return:
    """

    kernel = np.fft.fft2(kernel.astype(np.float64))
    kernel = np.fft.ifft2(1 / kernel)
    return convolve(image, kernel.astype(np.float32))

    # return (np.real(
    #     np.fft.ifft2(
    #         np.fft.fft2(image.astype(np.float32) * 256) /
    #         np.fft.fft2(np.fft.ifftshift(kernel.astype(np.float32)))
    #     )
    # ) / 256).astype(np.uint8)


if __name__ == "__main__":
    keanu = cv2.imread("./data/keanu.jpg")

    keanu_sp = sp_noise(keanu)
    keanu_sp_clr = sp_noise(keanu, mode="colour")
    keanu_gauss = gauss_noise(keanu, 25.)
    keanu_gauss_strong = gauss_noise(keanu, 70.)
    cv2.imwrite("./data/results/keanu_sp.png", keanu_sp)
    cv2.imwrite("./data/results/keanu_sp_colour.png", keanu_sp_clr)
    cv2.imwrite("./data/results/keanu_gauss_sigma25.png", keanu_gauss)
    cv2.imwrite("./data/results/keanu_gauss_sigma70.png", keanu_gauss_strong)

    imshow("orig", keanu)
    imshow("sp", keanu_sp)
    imshow("sp colour", keanu_sp_clr)
    imshow("gauss", keanu_gauss)

    one_time = convolve(convolve(keanu_sp, kernel=(11, 11), preset="median"), kernel=(3, 3), preset="sharpen")
    keanu_sp_recovered = convolve(convolve(one_time, kernel=(3, 3), preset="median"), kernel=(3, 3), preset="sharpen")
    cv2.imwrite("./data/results/keanu_sp_11m_3s.png", one_time)
    cv2.imwrite("./data/results/keanu_sp_11m_3s_3m_3s.png", keanu_sp_recovered)
    imshow("sp_rec", keanu_sp_recovered)
    imshow("denoise",
           cv2.fastNlMeansDenoisingColored(keanu_sp, h=30, hColor=30, templateWindowSize=17, searchWindowSize=57))

    keanu_sp_clr_recovered = convolve(convolve(keanu_sp_clr, kernel=(11, 11), preset="median"), kernel=(3, 3), preset="sharpen")
    imshow("sp colour", keanu_sp_clr_recovered)

    keanu_gauss_recovered = convolve(convolve(keanu_gauss, kernel=(11, 11), preset="median"), kernel=(5, 5), preset="sharpen")
    imshow("gauss", keanu_gauss_recovered)

    recovered = keanu_gauss.copy()
    for i in range(5):
        recovered = convolve(recovered, kernel=(i * 2 + 1, i * 2 + 1), preset="median")
        recovered = convolve(recovered, kernel=(i * 2 + 1, i * 2 + 1), preset="sharpen")
    imshow("5 iters after gauss", recovered)
