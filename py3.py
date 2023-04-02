# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# image = cv2.imread('asset/a.jpeg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray'), plt.show()

# def convertToRGB(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# face = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# face_c = face.detectMultiScale(gray, scaleFactor = 1.2 , minNeighbors = 5);
# print('Faces foud : ', len(face_c))

# for(x_face, y_face, w_face, h_face) in face_c:
#     cv2.rectingle(image, (x_face, y_face) , (x_face + w_face , y_face + w_face) , (0, 255, 0) ,10)

# plt.imshow(convertToRGB(image)),plt.show()

#___________________________________________
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# image=cv2.imread('asset/People.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray,cmap='gray'),plt.show()

# def convertToRGB(image):
#     return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# face_coo = face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);print('Faces found :',len(face_coo))

# for(x_face,y_face,w_face,h_face) in face_coo:
#     cv2.rectangle(image,(x_face,y_face),(x_face+w_face,y_face+w_face),(0,255,0),10)

# plt.imshow(convertToRGB(image)),plt.show()

#___________________________________________
# import cv2

# face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# cape = cv2.VideoCapture('/---/---/---/---/IMG_7668.mp4')

# while True:
#     _,img = cape.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face.detecMultiScale(gray, 1.1, 4)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img, (x,y) , (x + w , y + h) , (255, 0, 0) ,5)
#     cv2.imshow('Video', img)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cape.release()

# import cv2

# face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# cape = cv2.VideoCapture('/---/---/---/---/IMG_7668.mp4')

# while True:
#     _,img =cape.read()
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face.detectMultiScale(gray,1.1,4)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
#     cv2.imshow('Video',img)

#     k=cv2.waitKey(30) & 0xff
#     if k==27:
#         break

# cape.release()
#___________________________________________
import numpy as np
import cv2
import matplotlib.pyplot as plt

image=cv2.imread('asset/photo1680383713.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray'),plt.show()

def convertToRGB(image):
  return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

face_coo = face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
print('Faces found :',len(face_coo))

for(x_face,y_face,w_face,h_face) in face_coo:
  cv2.rectangle(image,(x_face,y_face),(x_face+w_face,y_face+w_face),(0,255,0),10)

plt.imshow(convertToRGB(image)),plt.show()