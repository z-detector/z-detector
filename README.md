
## Z-detector

 Це комп"ютер віжин детектор який розпізнає знаки Z,V,O на відео чи зображеннях

<p align="center"> 
<img src="1_img.png" width = 40% />  <img src="2_img.png" width = 40% /> 
</p>



### 1. Як використовувати 
 * На операційній системі з ядром лінукс (Ubuntu, Debian, інші..) виконуємо в терміналі:
 ```
 pip install -r requirements.txt
 ```
   Щоб встановити необхідні пакети
 
 
 
 * далі в терміналі виконуємо:

```
python3 detect.py --cfg a1.cfg --weights a1.weights
```
   де файл навченої нейронної мережі a1.weights скачуємо звідси - [Link](https://www.dropbox.com/s/ne2uau2a85edn69/weights.zip?dl=0)
   
   (лінки через якийсь час стають не активні, якщо лінк не працює - запитуємо на пошту)




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

 [dropbox part1](https://www.dropbox.com/s/bnvdmq5v4bq7ei5/military_vehicles.zip?dl=0) 
 
 (лінки через якийсь час стають не активні, якщо лінк не працює - запитуємо на пошту)

 [labelImg](https://github.com/heartexlabs/labelImg)
 прога для конвертації між типами та розмітки даних
 
### 5. Хибні спрацювання

Строка 

```
im0 = cv2.resize(img, (416,  416), interpolation=cv2.INTER_LINEAR)

```
стискає вхідне зображення до  416х416 - чим більша роздільна здатність - тим краще розпізнавання, але тим повільніше працює.
Необхідним є наявність відеокарти Nvidia на сервері чи комп"ютері на якому запускається алгоритм для його швидкої роботи. Це прискорює роботу приблизно сотню разів, залежить від моделі відеокарти. Бажано мати GTX1050, RTX2050, RTX3050 чи кращі. На таких картах можна запускати декілька камер одночасно 

Також чим більший поріг, тим менше хибних спрацюваннь.
parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')

 
### 6. Найближчі апдейти
 * Обновлення датасету
 * Обновлення моделі
 * Буде додана можливість запускати по списку камер
 * Інтерфейс збереження спрацюваннь
 
### 7. Питання, побажання, запити на співпрацю
Пишем сюди z_detector@yahoo.com

Можу допомогти з даними, навчанню та використанню нейронних мереж на хардвері
