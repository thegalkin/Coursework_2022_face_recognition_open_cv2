import cv2
import numpy as numpy
import pickle
import os

def main():

	# назначаем необходимые пути
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	cascade_loc = os.path.join(BASE_DIR,'cascades/data/haarcascade_frontalface_default.xml')

	# устанавливаем папку вывода результата обучения
	output_dir = os.path.join(BASE_DIR,'training_output')

	# инициализация каскадного классификатора по указанному пути
	faceCascade = cv2.CascadeClassifier(cascade_loc)

	# инициализация распознователя лиц по алгоритму "локальный бинарный паттерн"
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	# считываем данные лиц из обучения в распознователь
	recognizer.read(output_dir + '/trainner.yml')

	# создаем процесс записи видео
	cap = cv2.VideoCapture(0)

	# подгружаем имена персонажей
	labels = {}
	with open(output_dir + '/labels.pickle','rb') as f:
		org_labels = pickle.load(f)
		labels = {i:j for j,i in org_labels.items()}

	# безостановочно анализируем кадр за кадром, чтобы распозновать лица и обводить их квадратами с подписями
	while(True):

		# кадр за кадром
		ret, frame = cap.read() 

		# обучали в сером цвете, так и распознование в том же сером спектре
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		# ищем объекты разного размера на входном изображении. Найденные объекты возвращаются в виде списка прямоугольников.
		faces = faceCascade.detectMultiScale(
								gray,
								scaleFactor=1.5,
								minNeighbors=5
				)

		for x,y,w,h in faces: 
			# roi = region of interrest = регион обнаружения лица(мы его уже нашли)
			roi_gray = gray[y:y+h, x:x+w]  

			# распознание лица(лейбла) и получение оценки точности
			id_, conf = recognizer.predict(roi_gray)  
			print(conf, "% – ", labels[id_])
			# если оценка ниже/равна 60%, то выводим лейбл над лицом
			if conf <= 60:
				print(id_)
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255,255,255)
				stroke = 2
				cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

			# промежуточное сохранениние фото
			saved_image = 'my_image.png'
			cv2.imwrite(saved_image, roi_gray)

			# оформление квадратика
			color_square = (0,255,0) # BGR = blue green red - cv2 хранит цвета наоборот
			stroke = 2 				 
			coor_end_x = x+w
			coor_end_y = y+h
			cv2.rectangle(frame,(x,y),(coor_end_x,coor_end_y),color_square,stroke)


		cv2.imshow('Frame',frame)		# отображение получившегося квадратика

		# выхоид из распознователя по нажатию на q
		if cv2.waitKey(20) & 0xFF == ord('q'):
			break


	cap.release() # останавливает съемку 
	cv2.destroyAllWindows() # закрывает окно видоискателя
	
	print(" Спасибо за использование!")

try:
	main()
except KeyboardInterrupt:
	print(" Спасибо за использование!")
	exit(0)