from tkinter import Tk, Frame, Label, Canvas, Scrollbar, filedialog, BOTH, LEFT, RIGHT, Y, X, TOP, BOTTOM
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

image = None
processed_image = None

buttons = ["Load Image", "Save Image", "Gray Scale", "Resize", "Rotate", "Translate", "Gray Histogram", "RGB Histogram",
           "Histogram Equalization", "Contrast Stretching", "Thresholding", "Negative", "Log", "Power Law", 
           "Gaussian Blur", "Average Blur", "Median Blur", "Min Blur", "Max Blur", "Salt & Pepper", 
           "Gaussian Noise", "Sharpening Filters", "Edge Detection"]

root = Tk()
root.title("Image Processing App")
root.geometry("1280x720")

root.configure(bg='#FFFFFF')

main_frame = Frame(root, bg='#FFFFFF')
main_frame.pack(fill=BOTH, expand=True)

image_frame = Frame(main_frame, bg='#F5F5F5', relief='ridge', bd=2)
image_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)

footer_canvas = Canvas(main_frame, bg='#FFFFFF', height=80) 
footer_canvas.pack(side=BOTTOM, fill=X, padx=10, pady=5)

footer_scrollbar = Scrollbar(main_frame, orient='horizontal', command=footer_canvas.xview) 
footer_scrollbar.pack(side=BOTTOM, fill=X)

footer_frame = Frame(footer_canvas, bg='#FFFFFF') 
footer_frame.bind("<Configure>", lambda e: footer_canvas.configure(scrollregion=footer_canvas.bbox("all")))

footer_canvas.create_window((0, 0), window=footer_frame, anchor="nw") 
footer_canvas.configure(xscrollcommand=footer_scrollbar.set)

def display_placeholder():
    image_label = Label(image_frame, text='Image Display Area', bg='#F5F5F5', font=('Arial', 24), fg='#333333')
    image_label.pack(expand=True)
    return image_label

image_label = display_placeholder()

style = ttk.Style()
style.configure('TButton', 
                font=('Arial', 12), 
                padding=8, 
                relief='flat',  
                background='#4CAF50',  
                foreground='black')
style.map('TButton',
          background=[('active', '#81C784')],  
          foreground=[('active', 'black')])

def display_image(img, is_gray=False, scale=1.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (width, height))

    if is_gray:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(resized_img)
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img


def load_image():
    global image, processed_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        processed_image = image.copy() 
        display_image(processed_image)


def save_image():
    global processed_image
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG files", "*.png"),("JPEG files", "*.jpg"),("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, processed_image)

def gray_scale():
    global image, processed_image
    if image is not None:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_image(processed_image, is_gray=True)

def resize_image(width=300, height=200):
    global image, processed_image
    if image is not None:
        processed_image = cv2.resize(image, (width, height))
        display_image(processed_image, is_gray=False)

def rotate_image(angle=45):
    global image, processed_image
    if image is not None:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        processed_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        display_image(processed_image, is_gray=False)

def translate_image( tx=50, ty=50):
        global image, processed_image
        if image is not None:
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            h, w = image.shape[:2]
            processed_image = cv2.warpAffine(image, translation_matrix, (w, h))
            display_image(processed_image)

def gray_histogram():
    global image, processed_image
    if processed_image is not None:
        plt.hist(processed_image.ravel(), 256, [0, 256])
        plt.title("Gray Image Histogram")
        plt.show()

def rgb_histogram():
    global image, processed_image
    if image is not None:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("RGB Histogram")
        plt.show()

def histogram_equalization():
    global image, processed_image
    if processed_image is not None:
        processed_image = cv2.equalizeHist(processed_image)
        display_image(processed_image,)

def contrast_stretching():
    global image, processed_image
    if image is not None:
        min_val = np.min(image)
        max_val = np.max(image)
        processed_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        display_image(processed_image)

def thresholding( threshold=127):
    global image, processed_image
    if processed_image is not None:
        _, processed_image = cv2.threshold(processed_image, threshold, 255, cv2.THRESH_BINARY)
        display_image(processed_image)

def negative():
    global image, processed_image
    if image is not None:
        processed_image = 255 - image
        display_image(processed_image)
        
def log():
    global image, processed_image
    c =  255 / np.log(1 + np.max(image))
    log_trans = np.uint8( c * np.log(1+ image))
    display_image(log_trans)

def Power_law():
    global image, processed_image
    for gamma in [.1, .5, 1.2, 2.2]:
            norm_img = image / 255
            gamma_trans = np.uint8(np.array( 255 * (norm_img ** gamma)))
    display_image(gamma_trans)

def gaussian_blur( x=5, y=5):
    global image, processed_image
    if image is not None:
        processed_image = cv2.GaussianBlur(image, (x, y), 0)
        display_image(processed_image)

def average_blur( x=5, y=5):
    global image, processed_image
    if image is not None:
        processed_image = cv2.blur(image, (x, y), 0)
        display_image(processed_image)

def median_blur( median=5):
    global image, processed_image
    if image is not None:
        processed_image = cv2.medianBlur(image, median)
        display_image(processed_image)

def min_blur( x=5, y=5):
    global image, processed_image
    if image is not None:
        kernel = np.ones((x,y), np.uint8)
        processed_image = cv2.erode(image, kernel )
        display_image(processed_image)

def max_blur( x=5, y=5):
    global image, processed_image
    if image is not None:
        kernel = np.ones((x,y), np.uint8)
        processed_image = cv2.dilate(image, kernel )
        display_image(processed_image)

def salt_and_pepper( salt_prob=0.1, pepper_prob=0.1):
    global image, processed_image
    if image is not None:
        noisy_image = image.copy()
        row, col, ch = noisy_image.shape
        num_salt = np.ceil(salt_prob * row * col * ch)
        num_pepper = np.ceil(pepper_prob * row * col * ch)
        salt_coords = [np.random.randint(0, i, int(num_salt)) for i in noisy_image.shape]
        noisy_image[salt_coords[0], salt_coords[1], :] = 255
        pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in noisy_image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
        processed_image = noisy_image
        display_image(processed_image)

def gaussian_noise():
    global image, processed_image
    if image is not None:
        row, col, ch = image.shape
        mean = 0
        sigma = 10
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
        noisy_image = image.astype(np.float32) + gauss
        processed_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        display_image(processed_image)

def sharpening_filters():
    global image, processed_image
    if image is not None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv2.filter2D(image, -1, kernel)
        display_image(processed_image)

def sobel_edge_detection():
    global image, processed_image
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y).astype(np.uint8)
        processed_image = gradient_magnitude
        display_image(processed_image)

def canny_edge_detection( low_threshold=50, high_threshold=150):
    global image, processed_image
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        processed_image = edges
        display_image(processed_image)

def button_clicked(index):
    if index == 1:
        load_image()
    elif index == 2:
        save_image()
    elif index == 3:
        gray_scale()
    elif index == 4:
        resize_image()
    elif index == 5:
        rotate_image()
    elif index == 6:
        translate_image()
    elif index == 7:
        gray_histogram()
    elif index == 8:
        rgb_histogram()
    elif index == 9:
        histogram_equalization()
    elif index == 10:
        contrast_stretching()
    elif index == 11:
        thresholding()
    elif index == 12:
        log()
    elif index == 13:
        Power_law()
    elif index == 14:
        gaussian_blur()
    elif index == 15:
        average_blur()
    elif index == 16:
        median_blur()
    elif index == 17:
        min_blur()
    elif index == 18:
        max_blur()
    elif index == 19:
        salt_and_pepper()
    elif index == 20:
        gaussian_noise()
    elif index == 21:
        sharpening_filters()
    elif index == 22:
        sobel_edge_detection()
    elif index == 23:
        canny_edge_detection()    
    else:
        print(f"Button {index} clicked!")

for i, button_name in enumerate(buttons, start=1):
    button = ttk.Button(footer_frame, text=button_name, command=lambda i=i: button_clicked(i), style='TButton')
    button.pack(side=LEFT, padx=5, pady=5)

root.mainloop()
