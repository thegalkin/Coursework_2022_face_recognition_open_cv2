import cv2
import os
from PIL import Image
import numpy as np
import pickle

# назначаем необходимые пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')
cascade_loc = os.path.join(BASE_DIR,'cascades/data/haarcascade_frontalface_default.xml')
output_dir = os.path.join(BASE_DIR,'training_output')
# устанавливаем папку вывода результата обучения
try: 
	os.mkdir(output_dir)
except FileExistsError:
	print("output folder allready exists")

# инициализация каскадного классификатора по указанному пути
face_cascade = cv2.CascadeClassifier(cascade_loc)

# инициализация распознователя лиц по алгоритму "локальный бинарный паттерн"
recognizer = cv2.face.LBPHFaceRecognizer_create()

# подготовка переменных, в которых хранятся результаты обучения
current_id = 0
label_ids = {}  #dict 
y_labels = []
x_train = []

# проходим по всем файлам и папкам среди которых обучаем модель
for root,dirs,files in os.walk(image_dir):
	for file in files:
		# отсеиваем только картинки
		if file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
			path = os.path.join(root,file)
			label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()

			# если это новый для нас "персонаж", то создаем ему ячейку
			if label not in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			# превращаем картинку в массив numpy, используя python image library
			pil_image = Image.open(path).convert("L")  # "L" превращает цветную картинку в черно-белую
			
			# оптимизируем картинку для изменения размеров
			size = (550,550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)

			image_array = np.array(final_image, 'uint8')  # uint8 неподписанный int от 0 до 255 и переводим его в массив numpy
			# ищем объекты разного размера на входном изображении. Найденные объекты возвращаются в виде списка прямоугольников.
			faces = face_cascade.detectMultiScale(image_array) 
			

			for (x,y,w,h) in faces:
				# roi = region of interrest = регион обнаружения лица(мы его уже нашли)
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)   # откладываем данные для обучения 
				y_labels.append(id_)   # храним каждое лицо с приписанным ему лейблом 

# переключаемся на папку вывода перед началом обучения 
os.chdir(output_dir)
with open("labels.pickle", 'wb') as f: # подготовка к записи байтов в виде файла
	print("сохраняем персонажей...", label_ids)
	pickle.dump(label_ids,f)   # сохраняем лейблы для будущего параллельного использования с базой данных по лицам, на основе общего с ней id

# непосредственно обучение собранными данными
recognizer.train(x_train,np.array(y_labels)) 
recognizer.save("trainner.yml")