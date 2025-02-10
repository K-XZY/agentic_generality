from detect_text.Text import Text
import numpy as np
import cv2
import json
import time
import os
from os.path import join as pjoin


def save_detection_json(file_path, texts, img_shape):
    f_out = open(file_path, 'w')
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = loc['left'], loc['top'], loc['right'], loc['bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)
    json.dump(output, f_out, indent=4)


def visualize_texts(org_img, texts, shown_resize_height=None, show=False, write_path=None):
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)

    img_resize = img
    if shown_resize_height is not None:
        img_resize = cv2.resize(img, (int(shown_resize_height * (img.shape[1]/img.shape[0])), shown_resize_height))

    if show:
        cv2.imshow('texts', img_resize)
        cv2.waitKey(0)
        cv2.destroyWindow('texts')
    if write_path is not None:
        cv2.imwrite(write_path, img)


def text_sentences_recognition(texts):
    '''
    Merge separate words detected by Google ocr into a sentence
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(text_b, 'h', bias_justify=0.2 * min(text_a.height, text_b.height), bias_gap=2 * max(text_a.word_width, text_b.word_width)):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def merge_intersected_texts(texts):
    '''
    Merge intersected texts (sentences or words)
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_intersected(text_b, bias=2):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts




def text_cvt_orc_format_easyocr(result):
    texts = []
    for i, line in enumerate(result):
        # 'line[0]' is the list of 4 corner points of the bounding box.
        points = np.array(line[0])
        # Compute the bounding box (location) as the min/max of the x and y coordinates.
        location = {
            'left': int(np.min(points[:, 0])),
            'top': int(np.min(points[:, 1])),
            'right': int(np.max(points[:, 0])),
            'bottom': int(np.max(points[:, 1]))
        }
        # In EasyOCR, the recognized text is returned directly as the second element.
        content = line[1]
        # Append the text object (assuming Text is defined elsewhere)
        texts.append(Text(i, content, location))
    return texts


def text_filter_noise(texts):
    valid_texts = []
    for i, line in enumerate(texts):
        if len(line[1]) <= 1 and line[1].lower() not in ['a', ',', '.', '!', '?', '$', '%', ':', '&', '+']:
            continue
        valid_texts.append(line[1])
    return valid_texts


def text_detection(input_file='../data/input/30800.jpg', output_file='../data/output', show=False, method='google', paddle_model=None):
    '''
    :param method: google or paddle
    :param paddle_model: the preload paddle model for paddle ocr
    '''
    start = time.process_time()
    name = input_file.split('/')[-1][:-4]
    ocr_root = pjoin(output_file, 'ocr')
    img = cv2.imread(input_file)

    if method == 'easyocr':
        from PIL import Image
        import easyocr

        print('*** Detect Text through EasyOCR ***')
        easyocr_reader = easyocr.Reader(['en'], gpu=False)
        result = easyocr_reader.readtext(input_file)
        texts = text_cvt_orc_format_easyocr(result)


    else:
        raise ValueError('Method has to be easyocr')

    visualize_texts(img, texts, shown_resize_height=800, show=show, write_path=pjoin(ocr_root, name+'.png'))
    
    result = easyocr_reader.readtext(input_file)
    result = text_filter_noise(result)
    for detection in result:
        # detection[1] is the recognized text
        recognized_text = detection[1]
        print(recognized_text)


    
    save_detection_json(pjoin(ocr_root, name+'.json'), texts, img.shape)
    print("[Text Detection Completed in %.3f s] Input: %s Output: %s" % (time.process_time() - start, input_file, pjoin(ocr_root, name+'.json')))



