# DolphinDetection
 A project for dolphin detection based online video stream


# Project Structure

```
├── data            // Will be cleaned and re-created every runtime        
│   └── candidates  // Save the detected sample
        ├── 0.png
│       ├── 1.png
│       └── ...
│   └── frames      // Save the video stream as a single frame
        ├── 0       // Frames from No.1 video monitor
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           └── ...
│       ├── 1       // Video Stream from No.2 video monitor
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           └── ...
│       └── ...
│   └── videos      // Buffered video stream
│       ├── 0       // Video Stream from No.1 video monitor
│           ├── 000.ts
│           ├── 001.ts
│           ├── 002.ts
│           └── ...
│       ├── 1       // Video Stream from No.2 video monitor
│           ├── 000.ts
│           ├── 001.ts
│           ├── 002.ts
│           └── ...
│       └── ...
├── detection       // Module of building object detection
│   └── detection.py
│   └── thresh.py
│   └── ...
├── stream           // Module of reading video stream
│   └── frame.py
│   └── video.py
│   └── ...
├── interface       // Module of service interfaces
│   └── interface.py
│   └── ...
├── log             // Save system logs
│   └── interface.py
│   └── ...
├──  vcfg           // Confiurations of all videos
│   └── video.py
│   └── video_template.py
│   └── ...
├──  test.py        // Includes all test scripts and test cases of service interface
├──  config.py      // Unified constants and path definitions in the whole project
```

# Testing




