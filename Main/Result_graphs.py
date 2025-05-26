from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# open method used to open different extension image file
im1 = Image.open(r"285920\Result_graphs\energy_1")
im2 = Image.open(r"285920\Result_graphs\energy_2")
im3 = Image.open(r"285920\Result_graphs\energy_3")
im4 = Image.open(r"285920\Result_graphs\th_1")
im5 = Image.open(r"285920\Result_graphs\th_2")
im6 = Image.open(r"285920\Result_graphs\th_3")

# This method will show image in any image viewer
im1.show()
im2.show()
im3.show()
im4.show()
im5.show()
im6.show()


