from transformers import CLIPProcessor, CLIPModel

def load_clip_model():
    model_path = "./models/clip-vit-large-patch14"  # 模型路径
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    print("CLIP 模型加载成功！")
    return model, processor

# 测试加载
if __name__ == "__main__":
    load_clip_model()
