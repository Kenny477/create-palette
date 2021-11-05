import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
from sklearn.cluster import KMeans

class Palette:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.IMAGE = image
        self.CLUSTERS = clusters
    
    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def dominantColors(self):
    
        #read image
        img = cv2.imread(self.IMAGE)
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
    def plotClusters(self):
        img = cv2.imread(self.IMAGE)
        #convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #get rgb values from image to 1D array
        r, g, b = cv2.split(img)
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()

        #plotting 
        fig = plt.figure()
        ax = Axes3D(fig)
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        plt.show()
    
    def drawPalette(palette: list, size: int) -> Image.Image:
        img = Image.new("RGB", (size*len(palette), size))
        for index, color in enumerate(palette):
            for i in range(size):
                for j in range(size):
                    img.putpixel((i+size*index, j), color)
        return img



img = 'img1.jpg'
clusters = 10

p = Palette(img, clusters) 
colors = p.dominantColors()
print(colors)


