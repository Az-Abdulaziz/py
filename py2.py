# import cv2
# import numpy as np 
# import matplotlib.pyplot as plt
# import requests
# from PIL import Image

# img = Image.open(requests.get("https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg", stream= True).raw)
# img = img.resize((450,250))
# img_arr = np.array(img)

# gray = cv2.cvtColor(img_arr , cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5,5) ,0)
# Image.fromarray(blur)
# dilated = cv2.dilate(blur, np.ones((3,3)))
# Image.fromarray(dilated)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
# Image.fromarray(closing)

# Car_SRC = "/--/--/--/--/asset/haarcascade_car.xml"
# car = cv2.CascadeClassifier(Car_SRC)
# cars = car.detectMultiScale(closing, 1.1, 1)

# cnt = 0
# for(x,y,w,h) in cars:
#     cv2.rectangle(img_arr,(x,y) , (x+w ,y+h),(255 , 0 , 0) ,2)
#     cnt += 1
#     print(cnt, "Cars")
#     Image.fromarray(img_arr)

# cv2.imshow('image', img_arr)
# cv2.waitKey()
# cv2.destroyAllWindows()


#_______________________________________
# import cv2
# import numpy as np 
# import matplotlib.pyplot as plt

# img = cv2.imread("asset/img.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# des = cv2.cornerHarris(gray , 2 , 5 , 0.07)
# des = cv2.dilate(des, None)

# img[des > 0.01 * des.max()]=[255, 0, 0]
# plt.imshow(img)
# plt.waitforbuttonpress()

#_______________________________________
# import cv2
# import numpy as np 
# import matplotlib.pyplot as plt

# img = cv2.imread("asset/img.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(img),plt.show()

# corner = cv2.goodFeaturesToTrack(gray,30, 0.01 ,10)
# corner = np.int0(corner)

# for i in corner:
#     x, y = i.ravel()
#     cv2.circle(img, (x,y), 3, 255, -1)

# plt.imshow(img)
# plt.waitforbuttonpress()

# rgp = cv2.cvColor(imge, cv2.COLOR_BGR2RGB)
# plt.imshow(imge)
# plt.waitforbuttonpress()

#_______________________________________
# from PIL import Image
# import cv2
# import numpy as np
# import requests

# image = Image.open(requests.get(''))
# image = image.resize((450,250))
# image_arr = np.array(image)

# grey = cv2.cvColor(image_arr, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(grey, (5,5), 0)
# Image.fromarray(blur)

# dilated = cv2.dilate(blur, np.ones((3,3)))
# Image.fromarray(dilated)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# closing = cv2.morphologyEX(dilated, cv2.MORPH_CLOSE, kernel)
# Image.fromarray(closing)

# car_src = ''
# car = cv2.CascadeClassifier(car_src)
# cars = car.detectMultiScale(closing, 1.1, 1)

# cnt = 0
# for(x,y,w,h) in cars:
#     cv2.rectangle(image_arr, (x,y) , (x+w , y+h) , (255, 0, 0) ,2)
#     cnt += 1
#     print(cnt, "CARS")
#     Image.fromarray(image_arr)

# cv2.imshow('image' , image_arr)
# cv2.waitKey()
# cv2.destroyAllWindows()