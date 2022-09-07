
## Z-detector

 Це комп"ютер віжин детектор який розпізнає знаки Z,V,O на відео чи зображеннях

<p align="center"> 
<img src="1_img.png" width = 40% />  <img src="2_img.png" width = 40% /> 
</p>



### 1. Як використовувати 
 В терміналі виконуємо:

```
python3 detect.py --cfg /home/user/yolov3.cfg --weights /home/user/yolov3_last.weights
```
 де yolov3_last.weights тут - https://easyupload.io/1vngmj

### 2. Як підключити камеру
 В строку наведену нижче - пишем

```
cap1 = cv2.VideoCapture("rtsp://192.168.1.2:8080/out.h264")
```
де rtsp://192.168.1.2:8080/out.h264 адреса вашої камери

### 3. Як тренувати

 https://github.com/AlexeyAB/darknet

### 4. Датасет
 
 Датасет взятий з відкритих джерел, а саме з ютубу. 
 
 Список знаходиться в файлі list.txt
 
 Лінк на датасет:

 https://easyupload.io/hihog3
 
### Найближчі апдейти
 * Обновлення датасету
 * Обновлення моделі
 * Буде додана можливість запускати по списку камер
 * Інтерфейс збереження спрацюваннь
 
 
