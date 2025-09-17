# File: src/test_yolo.py
from ultralytics import YOLO
import cv2
import numpy as np

def test_yolo_installation():
    """Test if YOLO is working properly"""
    print("Testing YOLO installation...")
    
    try:
        # Load YOLO model (this will download it first time)
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')  # Downloads ~6MB model
        print("✅ YOLO model loaded successfully!")
        
        # Create a dummy image with a person-like shape for testing
        print("Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple rectangle to simulate a person
        cv2.rectangle(test_image, (250, 150), (350, 400), (100, 100, 100), -1)
        
        # Test detection
        print("Running detection test...")
        results = model(test_image)
        
        print("✅ YOLO detection completed!")
        print(f"Detected {len(results[0].boxes)} objects")
        
        # Print detected classes
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"Detected: {class_name} (confidence: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing YOLO: {e}")
        return False

def test_opencv():
    """Test OpenCV installation"""
    print("\nTesting OpenCV...")
    try:
        # Test basic OpenCV functions
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV working correctly!")
        return True
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False

if __name__ == "__main__":
    print("=== AI Footstep Generator - Installation Test ===\n")
    
    # Test all components
    opencv_ok = test_opencv()
    yolo_ok = test_yolo_installation()
    
    print("\n=== Test Results ===")
    print(f"OpenCV: {'✅ PASS' if opencv_ok else '❌ FAIL'}")
    print(f"YOLO: {'✅ PASS' if yolo_ok else '❌ FAIL'}")
    
    if opencv_ok and yolo_ok:
        print("\n🎉 All tests passed! Ready to start development.")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")