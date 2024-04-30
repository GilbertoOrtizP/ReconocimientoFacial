import cv2
import face_recognition
import matplotlib.pyplot as plt

image1=cv2.imread("ImagenesRostros/Brad Pitt/Brad Pitt Rotado.jpg")
image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image2=cv2.imread("ImagenesRostros/Brad Pitt/Imagen2.jpg")
image2= cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

face_loc1=face_recognition.face_locations(image1)
if face_loc1 != []:
    vector_rostro1 = face_recognition.face_encodings(image1, known_face_locations=[face_loc1][0])
    vector_rostro1=vector_rostro1[0]

    face_loc2=face_recognition.face_locations(image2)
    vector_rostro2 = face_recognition.face_encodings(image2, known_face_locations=[face_loc2][0])
    vector_rostro2=vector_rostro2[0]

    resultado = face_recognition.compare_faces([vector_rostro1],vector_rostro2)
    print(resultado)
else:
    angulo=0
    ancho=image1.shape[1]
    alto=image1.shape[0]
    while face_loc1 !=[] and angulo < 360:
        Mrot=cv2.getRotationMatrix2D((ancho//2,alto//2),angulo,1)
        image1rot=cv2.warpAffine(image1,Mrot,(ancho,alto))
        face_loc1=face_recognition.face_locations(image1rot)
        angulo=angulo+10
    if face_loc1==[]:
        print("No hay rostros en la imagen")
    else:
        vector_rostro1 = face_recognition.face_encodings(image1, known_face_locations=[face_loc1][0])
        vector_rostro1=vector_rostro1[0]

        face_loc2=face_recognition.face_locations(image2)
        vector_rostro2 = face_recognition.face_encodings(image2, known_face_locations=[face_loc2][0])
        vector_rostro2=vector_rostro2[0]

        resultado = face_recognition.compare_faces([vector_rostro1],vector_rostro2)
        print(resultado)



