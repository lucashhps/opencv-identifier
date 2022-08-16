import cv2
import numpy as np

# essas funções são usadas para evitar que, por exemplo, o valor de low hue seja maior do que o valor do high hue


class trackbarIT:

    def __init__(self, mask_name):
        self.window_detection_name = mask_name
        self.max_value = 255
        self.max_value_H = 180
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = self.max_value_H
        self.high_S = self.max_value
        self.high_V = self.max_value
        self.low_H_name = 'L-H'
        self.low_S_name = 'L-S'
        self.low_V_name = 'L-V'
        self.high_H_name = 'U-H'
        self.high_S_name = 'U-S'
        self.high_V_name = 'U-V'
        
    def on_low_H_thresh_trackbar(self, val):
        #global low_H
        #global high_H
        self.low_H = val
        self.low_H = min(self.high_H-1, self.low_H)
        cv2.setTrackbarPos(self.low_H_name, self.window_detection_name, self.low_H)
    def on_high_H_thresh_trackbar(self, val):
        #global low_H
        #global high_H
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H+1)
        cv2.setTrackbarPos(self.high_H_name, self.window_detection_name, self.high_H)
    def on_low_S_thresh_trackbar(self, val):
        #global low_S
        #global high_S
        self.low_S = val
        self.low_S = min(self.high_S-1, self.low_S)
        cv2.setTrackbarPos(self.low_S_name, self.window_detection_name, self.low_S)
    def on_high_S_thresh_trackbar(self, val):
        #global low_S
        #global high_S
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S+1)
        cv2.setTrackbarPos(self.high_S_name, self.window_detection_name, self.high_S)
    def on_low_V_thresh_trackbar(self, val):
        #global low_V
        #global high_V
        self.low_V = val
        self.low_V = min(self.high_V-1, self.low_V)
        cv2.setTrackbarPos(self.low_V_name, self.window_detection_name, self.low_V)
    def on_high_V_thresh_trackbar(self, val):
        #global low_V
        #global high_V
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V+1)
        cv2.setTrackbarPos(self.high_V_name, self.window_detection_name, self.high_V)
        
    def create_tracks(self):
        cv2.namedWindow(self.window_detection_name)
        cv2.createTrackbar(self.low_H_name, self.window_detection_name , self.low_H, self.max_value_H, trackbarIT(self.window_detection_name).on_low_H_thresh_trackbar) # lower hue
        cv2.createTrackbar(self.low_S_name, self.window_detection_name , self.low_S, self.max_value, trackbarIT(self.window_detection_name).on_low_S_thresh_trackbar) # lower saturation
        cv2.createTrackbar(self.low_V_name, self.window_detection_name , self.low_V, self.max_value, trackbarIT(self.window_detection_name).on_low_V_thresh_trackbar) # lower value
        cv2.createTrackbar(self.high_H_name, self.window_detection_name , self.high_H, self.max_value_H, trackbarIT(self.window_detection_name).on_high_H_thresh_trackbar) # upper hue
        cv2.createTrackbar(self.high_S_name, self.window_detection_name , self.high_S, self.max_value, trackbarIT(self.window_detection_name).on_high_S_thresh_trackbar) # upper saturation
        cv2.createTrackbar(self.high_V_name, self.window_detection_name , self.high_V, self.max_value, trackbarIT(self.window_detection_name).on_high_V_thresh_trackbar) # upper value

    def update_par(self): # atualiza os valores de cada parametro de cada instância
        self.low_H = cv2.getTrackbarPos('L-H' ,trackbar.window_detection_name)
        self.low_S = cv2.getTrackbarPos('L-S' ,trackbar.window_detection_name)
        self.low_V = cv2.getTrackbarPos('L-V' ,trackbar.window_detection_name)
        self.high_H = cv2.getTrackbarPos('U-H' ,trackbar.window_detection_name)
        self.high_S = cv2.getTrackbarPos('U-S' ,trackbar.window_detection_name)
        self.high_V = cv2.getTrackbarPos('U-V' ,trackbar.window_detection_name)

    def create_mask(self, frame): # cria uma máscara utilizando os parametros da 
        self.mask = cv2.inRange(frame, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V)) # define a máscara usando a img em hsv e aplicando os limites de hue, saturation e value
        kernel = np.ones((5,5), np.uint8)
        self.mask = cv2.erode(self.mask, kernel) # usa um quadrado kernel de 5x5 pra eliminar um pouco de noise da imagem
        return self.mask
    
trackbars = []
for color in ['red', 'blue', 'green']: 
    track = trackbarIT(color)
    track.create_tracks()
    trackbars.append(track)

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        allColors = {}
        for trackbar in trackbars:
            trackbar.update_par()
            mask = trackbar.create_mask(hsv)

            # Contours detection
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # a variavel "_" é a variável das hierarquias dos contornos, mas ela n foi usada, por isso é só um "_"
            allColors[trackbar.window_detection_name] = contours
        
        for color in allColors:
            for cnt in allColors[color]:
                area = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) # os contornos ficavam com linhas meio tortas, por isso uso a aproximação pra deixar elas retas
                x = approx.ravel()[0] # pega o x do primeiro ponto do contorno
                y = approx.ravel()[1] # pega o y do primeiro ponto do contorno
                if area > 400: # usa uma area > 400 como outra forma de eliminar noise da img
                    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5) # desenha os contornos aproximados

                    cv2.putText(frame, f'Color: {color}', (x, y-90), font, 1, (0, 0, 0))
                    cv2.putText(frame, f'Points: {len(approx)}', (x, y-50), font, 1, (0, 0, 0))
                    cv2.putText(frame, f'Area: {area}', (x, y-10), font, 1, (0, 0, 0))


        cv2.imshow('frame', frame)

        
        for track in trackbars:
            
            cv2.imshow(track.window_detection_name, track.mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

