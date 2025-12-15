from image_processor import CatImageProcessor

if __name__ == "__main__":

    processor = CatImageProcessor(output_dir="images")
    

    processor.run(limit=5)