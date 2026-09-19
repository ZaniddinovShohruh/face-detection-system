import os
import cv2
import numpy as np

K = 3
MAX_DIST = 0.24
CELLS = 8
BINS = 16
HOG_LEN = 1764

cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

folder = os.path.dirname(os.path.abspath(__file__))
photo_path = os.path.join(folder, "rasm.jpg")


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def shrink(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def get_boxes(gray):
    eq = cv2.equalizeHist(gray)
    h, w = eq.shape
    min_size = max(50, min(h, w) // 12)
    a = cascade1.detectMultiScale(eq, 1.1, 5, minSize=(min_size, min_size))
    b = cascade2.detectMultiScale(eq, 1.1, 5, minSize=(min_size, min_size))
    faces = list(a) + list(b)
    if len(faces) == 0:
        return []
    picked = []
    for (x, y, fw, fh) in faces:
        keep = True
        for (px, py, pw, ph) in picked:
            ix = max(x, px)
            iy = max(y, py)
            ax = min(x + fw, px + pw)
            ay = min(y + fh, py + ph)
            if ax > ix and ay > iy:
                inter = (ax - ix) * (ay - iy)
                if inter / float(fw * fh) > 0.4:
                    keep = False
                    break
        if keep:
            picked.append((int(x), int(y), int(fw), int(fh)))
    return picked


def crop_face(gray, box):
    x, y, w, h = box
    x = max(0, x + int(w * 0.08))
    y = max(0, y + int(h * 0.08))
    w = max(1, int(w * 0.84))
    h = max(1, int(h * 0.84))
    return gray[y:y + h, x:x + w]


def lbp_map(img):
    c = img[1:-1, 1:-1]
    code = np.zeros(c.shape, dtype=np.uint8)
    code |= (img[0:-2, 0:-2] >= c).astype(np.uint8) << 7
    code |= (img[0:-2, 1:-1] >= c).astype(np.uint8) << 6
    code |= (img[0:-2, 2:] >= c).astype(np.uint8) << 5
    code |= (img[1:-1, 2:] >= c).astype(np.uint8) << 4
    code |= (img[2:, 2:] >= c).astype(np.uint8) << 3
    code |= (img[2:, 1:-1] >= c).astype(np.uint8) << 2
    code |= (img[2:, 0:-2] >= c).astype(np.uint8) << 1
    code |= (img[1:-1, 0:-2] >= c).astype(np.uint8)
    return code


def lbp_vec(face):
    face = cv2.resize(face, (96, 96))
    code = lbp_map(face)
    ch, cw = code.shape
    parts = []
    for i in range(CELLS):
        for j in range(CELLS):
            y0 = i * ch // CELLS
            y1 = (i + 1) * ch // CELLS
            x0 = j * cw // CELLS
            x1 = (j + 1) * cw // CELLS
            hist, _ = np.histogram(code[y0:y1, x0:x1], bins=BINS, range=(0, 256), density=True)
            parts.append(hist.astype(np.float32))
    vec = np.concatenate(parts)
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def hog_vec(face):
    empty = np.zeros(HOG_LEN, dtype=np.float32)
    if face is None or face.size == 0:
        return empty
    img = cv2.resize(face, (64, 64))
    v = hog.compute(img)
    if v is None:
        return empty
    v = v.reshape(-1).astype(np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def face_to_vector(gray, box):
    face = crop_face(gray, box)
    if face.size == 0:
        return np.zeros(HOG_LEN + CELLS * CELLS * BINS, dtype=np.float32)
    face = cv2.resize(face, (96, 96))
    face = cv2.equalizeHist(face)
    return vector_from_face(face)


def vector_from_face(face):
    vec = np.concatenate([hog_vec(face), lbp_vec(face)])
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def extra_faces(face):
    h, w = face.shape
    items = [face]
    items.append(cv2.convertScaleAbs(face, alpha=1.15, beta=12))
    items.append(cv2.convertScaleAbs(face, alpha=0.85, beta=-12))
    items.append(cv2.GaussianBlur(face, (3, 3), 0))
    for dx, dy in ((5, 0), (-5, 0), (0, 4), (0, -4)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        items.append(cv2.warpAffine(face, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return items


def add_known_photo(path, train_vecs, train_labels):
    img = load_image(path)
    if img is None:
        return 0
    img = shrink(img, 800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = get_boxes(gray)
    if len(boxes) == 0:
        return 0
    box = max(boxes, key=lambda b: b[2] * b[3])
    face = crop_face(gray, box)
    if face.size == 0:
        return 0
    face = cv2.resize(face, (96, 96))
    face = cv2.equalizeHist(face)
    n = 0
    for f in extra_faces(face):
        train_vecs.append(vector_from_face(f))
        train_labels.append("Known")
        n += 1
    return n


def knn(test_vec, train_vecs, train_labels, k):
    k = min(k, len(train_vecs))
    distances = []
    for i in range(len(train_vecs)):
        dist = 1.0 - float(np.dot(test_vec, train_vecs[i]))
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda item: item[0])
    nearest = distances[:k]
    votes = {}
    total = 0.0
    for dist, label in nearest:
        total += dist
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    best_label = None
    best_count = -1
    for label in votes:
        if votes[label] > best_count:
            best_count = votes[label]
            best_label = label
    mean_dist = total / float(k)
    return best_label, mean_dist, best_count


if not os.path.exists(photo_path):
    print("Put the face photo into rasm.jpg")
    raise SystemExit()

train_vecs = []
train_labels = []
added = add_known_photo(photo_path, train_vecs, train_labels)
if added == 0:
    print("No face found in rasm.jpg. Use a clear front photo.")
    raise SystemExit()

for name in os.listdir(folder):
    low = name.lower()
    if not low.endswith(".jpg"):
        continue
    if low == "rasm.jpg":
        continue
    add_known_photo(os.path.join(folder, name), train_vecs, train_labels)

print("Neighbors:", len(train_vecs), "  k =", K)
print("Click the camera window, then press Q or ESC.")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not camera.isOpened():
    print("Camera did not open")
    raise SystemExit()

win = "Face Detection"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = shrink(frame, 640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = get_boxes(gray)

        for (x, y, w, h) in boxes:
            vec = face_to_vector(gray, (x, y, w, h))
            label, dist, votes = knn(vec, train_vecs, train_labels, K)
            if label == "Known" and votes > K // 2 and dist <= MAX_DIST:
                text = "Known"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text + " " + str(round(dist, 2)), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "Faces: " + str(len(boxes)), (10, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, frame)

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
