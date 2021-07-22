
from imutils import face_utils
import dlib
import cv2
from playsound import playsound
from scipy.spatial import distance as dest
from threading import Thread
import time 
import serial



cap = cv2.VideoCapture(1)
p_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p_path)

closing_counter = 0
yawn_counter = 0
yawn_state = False
blinking_state = False
blinking_counter = 0
alarm_on = False
hart_state = False


ser = serial.Serial('COM3',baudrate = 9600,timeout = 1)
time.sleep(3)

class bpm :
  hartbeat = 0

def alarm():
  while True :
    if alarm_on == True :
      playsound("alarm.wav")
    else :
      break

def reader(): 
  while hart_state == True :
    bpm.hartbeat =ser.readline().decode('ascii')
    if bpm.hartbeat != '' and int(bpm.hartbeat) < 120 and int(bpm.hartbeat) > -1:
      bpm.hartbeat = int(bpm.hartbeat)
    else: 
      bpm.hartbeat = 0

def eyes_ar(eye , landmarks):
      p0 = landmarks.part(eye[0]).x , landmarks.part(eye[0]).y
      p1 = landmarks.part(eye[1]).x , landmarks.part(eye[1]).y
      p2 = landmarks.part(eye[2]).x , landmarks.part(eye[2]).y
      p3 = landmarks.part(eye[3]).x , landmarks.part(eye[3]).y
      p4 = landmarks.part(eye[4]).x , landmarks.part(eye[4]).y
      p5 = landmarks.part(eye[5]).x , landmarks.part(eye[5]).y

      hor_line = dest.euclidean(p0 ,p3)
      ver_line_1 = dest.euclidean(p1,p5)
      ver_line_2 = dest.euclidean(p2,p4)

      ear = (ver_line_1 + ver_line_2)/(2.0 * hor_line)

      return round(ear,4)


def mouth_ar(mouth , landmarks):
      p0 = landmarks.part(mouth[0]).x , landmarks.part(mouth[0]).y
      p6 = landmarks.part(mouth[6]).x , landmarks.part(mouth[6]).y


      p2 = landmarks.part(mouth[2]).x , landmarks.part(mouth[2]).y
      p10 = landmarks.part(mouth[10]).x , landmarks.part(mouth[10]).y

      p4 = landmarks.part(mouth[4]).x , landmarks.part(mouth[4]).y
      p8 = landmarks.part(mouth[8]).x , landmarks.part(mouth[8]).y

      hor_line = dest.euclidean(p0 ,p6)
      ver_line_1 = dest.euclidean(p2,p10)
      ver_line_2 = dest.euclidean(p4,p8)

      mar = (ver_line_1 + ver_line_2)/(2.0 * hor_line)
      return mar

while True: 
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  prev_yawn_status = yawn_state
  prev_blinking_status = blinking_state
  faces = detector(gray)

  for face in faces:
    x , y = face.left() , face.top()
    x1,y1 = face.right() , face.bottom()
    cv2.rectangle(frame,(x,y),(x1,y1) , (255,0,0) , 1)

    landmarks = predictor(gray , face)
    left_eye = range(36,42)
    right_eye = range(42,48)
    mouth = range(48,68)
      
    ratio_right = eyes_ar(right_eye,landmarks)
    ratio_left = eyes_ar(left_eye,landmarks)
    ear = ((ratio_right+ratio_left)/2)
    mar = mouth_ar(mouth,landmarks)
    if hart_state == False :
      hart_state = True
      t1 = Thread(target=reader)
      t1.start()  
    if 0.25 > ear or  (bpm.hartbeat < 40 and bpm.hartbeat > 30):
      closing_counter += 1
      blinking_state=True
      cv2.putText(frame, "eyes closed", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,255), 1)
      if closing_counter > 48 :
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0 ,255), 2)
        if alarm_on == False :
          alarm_on = True
          t = Thread(target=alarm)
          t.start()
        
        
    else :
      closing_counter = 0
      cv2.putText(frame, "eyes open", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 1)
      blinking_state=False 
      alarm_on = False

    if prev_blinking_status == True and blinking_state == False:
        blinking_counter+=1


    if mar > 0.75 :
      yawn_state = True
      
    else:
  		  yawn_state = False

    if prev_yawn_status == True and yawn_state == False:
        yawn_counter+=1


    output_text = "Yawn Count: " + str(yawn_counter)
    cv2.putText(frame, output_text, (450,450),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
    cv2.putText(frame,"blinking counter : {:}".format(blinking_counter),(10,25),
    cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)


    cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   
    cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "BPM: {:.2f}".format(bpm.hartbeat), (480, 90),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
   

  cv2.imshow('detection window', frame)
  if cv2.waitKey(1) & 0xff == ord('q'):
    hart_state = False
    break

cap.release()
cv2.destroyAllWindows()