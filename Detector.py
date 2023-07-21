import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Detector:
    def __int__(self):
        pass

    def readCalasses(self, classesFilePath):
        # Метод для чтения списка классов из текстового файла
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            # Создание списка случайных цветов для каждого класса
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

            print(len(self.classesList), len(self.colorList))

    def downloadModel(self, modelURL):
        # Метод для загрузки модели с заданного URL
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./Practice"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        # Метод для загрузки модели из каталога
        print("Загрузка модели " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Модель " + self.modelName + " загружена успешно...")

    def createBoundigBox(self, image, threshold=0.5):
        # Метод для создания ограничивающих рамок вокруг обнаруженных объектов на изображении
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        imH, imW, imC = image.shape

        bboxsIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                                iou_threshold=threshold, score_threshold=threshold)

        print(bboxsIdx)

        if len(bboxs) != 0:
            for i in bboxsIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {} % '.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                # Рисование прямоугольника вокруг обнаруженного объекта и отображение метки класса и уверенности
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                ########################
                # Рисование дополнительных линий на прямоугольнике для улучшения визуализации
                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))

                cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)

                ############################################################

                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

        return image

    def predictImage(self, imagePath, threshold=0.5):
        # Метод для обнаружения объектов на одном изображении
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundigBox(image, threshold)

        # Сохранение результата в файл с названием модели и отображение его
        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Результат", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold=0.5):
        # Метод для обнаружения объектов на видео
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Ошибка открытия файла…")
            return

        (success, image) = cap.read()
        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundigBox(image, threshold)

            # Отображение результатов на видео с указанием FPS
            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Результат", bboxImage)

            # Выход из цикла, если нажата клавиша 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()

        cv2.destroyAllWindows()
