# #https://github.com/RapidAI/RapidOCR
# from rapidocr import RapidOCR

# engine = RapidOCR()

# img_url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/resources/test_files/ch_en_num.jpg"
# result = engine(img_url)
# print(result)

# result.vis("vis_result.jpg")

# Initialize PaddleOCR instance
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False  # Disable OneDNN/MKLDNN
)

result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)

for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")