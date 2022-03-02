import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, request, make_response
from werkzeug.utils import secure_filename

from PIL import Image
import numpy as np
import base64
import io

import cv2
import imutils

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleandir(directo):
    for i in os.listdir(f'{directo}'):
        print(i)
        os.remove(f'{directo}/{i}')

def process(image):
	#image = cv2.imread('bg4.jpg') #reads the image
	dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
	#cv2.imwrite('dst.jpg',dst)
	
	

	
	#gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	#rgb_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
	#cv2.imwrite('ST_RGB_image.jpg',rgb_image)

	new_image = cv2.medianBlur(dst,5)
	#cv2.imwrite('median_blur.jpg',new_image)


	hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
	h, s, v = cv2.split(hsv_image)

	ycbcr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb)
	Y, Cr, Cb = cv2.split(ycbcr_image)

	#cv2.imwrite('H.jpg',h)

	ret, th1 = cv2.threshold(Cr,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	#cv2.imwrite('Binary_image.jpg',th1)

	kernel = np.ones((5,5), dtype = "uint8")/9
	bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
	erosion = cv2.erode(bilateral, kernel, iterations = 6)

	#cv2.imwrite('mask_erosion.jpg', erosion)


	

#find all your connected components (white blobs in your image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th1, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
	sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

	min_size = 7000

#your answer image
	img2 = np.zeros((output.shape))
	#for every component in the image, you keep it only if it's above min_size
	for i in range(0, nb_components):
		if sizes[i] >= min_size:
			img2[output == i + 1] = 255
			#cv2.imwrite('img2.jpg',img2)
		
	img3 = img2.astype(np.uint8) 
	#cv2.imwrite('binary_connected_components.jpg',img3)      
	# find contours in the thresholded image

	# find contours in the thresholded image
	cnts = cv2.findContours(img3.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	print("[INFO] {} unique contours found".format(len(cnts)))



	# loop over the contours
	for (i, c) in enumerate(cnts):
		# draw the contour
		((x, y), _) = cv2.minEnclosingCircle(c)
		cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	# show the output image
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)

	return image

	#cv2.imwrite('Result_BLue_Grape.jpg',image)

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='JPEG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():

	cleandir("static/uploads")
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		

		img = Image.open(app.config['UPLOAD_FOLDER'] + "/"+ filename)
		data = io.BytesIO()
		img.save(data, "JPEG")
		encode_img_data = base64.b64encode(data.getvalue())
		
		img1 = Image.open(app.config['UPLOAD_FOLDER'] + "/"+ filename)
		#img1 = Image.open(io.BytesIO(img1))
		npimg=np.array(img1)
		image1=npimg.copy()
		image1 =process(image1)

		np_img=Image.fromarray(image1)
		img_encoded=image_to_byte_array(np_img)  
		base64_bytes = base64.b64encode(img_encoded).decode("utf-8")
		
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=encode_img_data.decode("UTF-8"), filename1 = base64_bytes)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)	

	  

	

#@app.route('/display/<filename>')
#def display_image(filename):
#	#print('display_image filename: ' + filename)
#	return redirect(url_for('static', filename='uploads/' + filename), code=301)

	
if __name__ == "__main__":
    app.run()