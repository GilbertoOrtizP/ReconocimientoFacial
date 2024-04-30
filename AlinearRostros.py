import cv2
import face_recognition
import math

image1=cv2.imread("ImagenesRostros/Rostros Inclinados/Rostro5.jpg")
#image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
face_loc=face_recognition.face_locations(image1)
rasgos_loc1=face_recognition.face_landmarks(image1,model="small")
#print(face_loc[0][3])
face_loc1=face_loc[0]
rasgos_loc=rasgos_loc1[0]
#cv2.circle(image1,(455,144),5,(0,0,255),-1)

ojo_izq=rasgos_loc['left_eye'][0]
ojo_der=rasgos_loc['right_eye'][0]
cat_op=ojo_der[1]-ojo_izq[1]
cat_ady=ojo_der[0]-ojo_izq[0]
angulo_rad=math.atan2(cat_op,cat_ady)
angulo=math.degrees(angulo_rad)

#cv2.rectangle(image1, (face_loc1[3], face_loc1[0]), (face_loc1[1], face_loc1[2]),(155,0,255))
image_cut=image1[(face_loc1[0]-20):(face_loc1[2]+20),(face_loc1[3]-20):(face_loc1[1]+20)]
Mrot=cv2.getRotationMatrix2D((image_cut.shape[1]//2,image_cut.shape[0]//2),angulo,1)
image1rot=cv2.warpAffine(image_cut,Mrot,(image_cut.shape[1],image_cut.shape[0]))

cv2.imshow("Image",image1rot)
cv2.waitKey(0)
cv2.destroyAllWindows()