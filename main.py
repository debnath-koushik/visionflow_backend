from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import zipfile
import tempfile
import os
import shutil
import random
import cv2
from moviepy import *

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add your frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Matched-Files", "X-Extra-Labels-Removed", "X-Extra-Images-Removed", "X-Files-Added", "X-Train-Count", "X-Valid-Count", "X-Test-Count", "X-Total-Count", "X-Original-Duration", "X-Trimmed-Duration", "X-Total-Frames", "X-Extracted-Frames", "X-Frame-Gap", "X-FPS"],  # Expose custom headers
)


@app.post("/api/filter_image_by_labels/")
async def filter_image_by_labels(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Filter images by labels: keeps only files that have matching basenames
    between labels (.txt) and images (.jpg, .jpeg, .png)
    file1: labels ZIP file
    file2: images ZIP file
    """
    try:
        # Create a temporary working directory
        tmpdir = tempfile.mkdtemp()

        # Save uploaded ZIP files
        labels_zip_path = os.path.join(tmpdir, file1.filename)
        images_zip_path = os.path.join(tmpdir, file2.filename)
        
        with open(labels_zip_path, "wb") as f1:
            shutil.copyfileobj(file1.file, f1)
        with open(images_zip_path, "wb") as f2:
            shutil.copyfileobj(file2.file, f2)

        # Extract both zip files
        labels_dir = os.path.join(tmpdir, "labels_temp")
        images_dir = os.path.join(tmpdir, "images_temp")
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        with zipfile.ZipFile(labels_zip_path, 'r') as zip_ref:
            zip_ref.extractall(labels_dir)
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)

        # Helper: collect file basenames
        def list_files(base_dir, exts):
            files = set()
            for root, _, filenames in os.walk(base_dir):
                for f in filenames:
                    file_lower = f.lower()
                    # Check if file ends with any of the extensions
                    if any(file_lower.endswith(ext.lower()) for ext in exts):
                        files.add(os.path.splitext(f)[0])
            return files

        label_files = list_files(labels_dir, ('.txt',))
        image_files = list_files(images_dir, ('.jpg', '.jpeg', '.png'))

        # Common and extra files
        common_files = label_files.intersection(image_files)
        extra_labels = label_files - common_files
        extra_images = image_files - common_files

        # Check if we have any matching files
        if len(common_files) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No matching files found. Labels: {len(label_files)}, Images: {len(image_files)}. Files must have the same basename (e.g., 'image1.txt' and 'image1.jpg')."
            )

        # Remove extra files
        def clean_folder(base_dir, valid_bases, folder_type):
            removed_count = 0
            for root, _, filenames in os.walk(base_dir):
                for f in filenames:
                    base, _ = os.path.splitext(f)
                    file_path = os.path.join(root, f)
                    if base not in valid_bases:
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                removed_count += 1
                        except (PermissionError, OSError):
                            pass  # Skip if not a file or permission denied
            return removed_count

        clean_folder(labels_dir, common_files, "label")
        clean_folder(images_dir, common_files, "image")

        # Create the final zip with labels and images folders
        cleaned_zip_path = os.path.join(tmpdir, "cleaned_result.zip")
        
        # Count files to be added
        files_added = 0
        
        with zipfile.ZipFile(cleaned_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add labels folder - flatten structure to avoid nested folders
            for root, _, files in os.walk(labels_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    if os.path.isfile(abs_path):  # Ensure it's a file, not a directory
                        # Use only filename to avoid nested folder structure
                        arcname = os.path.join("labels", file)
                        zipf.write(abs_path, arcname)
                        files_added += 1
            # Add images folder - flatten structure to avoid nested folders
            for root, _, files in os.walk(images_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    if os.path.isfile(abs_path):  # Ensure it's a file, not a directory
                        # Use only filename to avoid nested folder structure
                        arcname = os.path.join("images", file)
                        zipf.write(abs_path, arcname)
                        files_added += 1

        # Ensure the zip file exists and has content
        if not os.path.exists(cleaned_zip_path):
            raise HTTPException(status_code=500, detail="Failed to create output ZIP file")
        
        zip_size = os.path.getsize(cleaned_zip_path)
        if zip_size == 0:
            raise HTTPException(status_code=500, detail="Output ZIP file is empty. No matching files found between labels and images.")

        # Read the zip file content into memory
        with open(cleaned_zip_path, 'rb') as f:
            zip_content = f.read()

        # Return ZIP file as streaming response with summary in headers
        def generate():
            yield zip_content

        response = StreamingResponse(
            generate(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="cleaned_result.zip"',
                "Content-Length": str(zip_size),
                "X-Matched-Files": str(len(common_files)),
                "X-Extra-Labels-Removed": str(len(extra_labels)),
                "X-Extra-Images-Removed": str(len(extra_images)),
                "X-Files-Added": str(files_added),
            }
        )
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train_test_valid_split/")
async def train_test_valid_split(file: UploadFile = File(...), train_ratio: int = Form(...), test_ratio: int = Form(...), valid_ratio: int = Form(...)):
    """
    Train test valid split: splits the dataset into train, test, and validation sets
    file: dataset ZIP file containing 'images' and 'labels' folders
    train_ratio: percentage of data for training
    test_ratio: percentage of data for testing
    valid_ratio: percentage of data for validation
    """
    try:
        # Validate total = 100%
        if train_ratio + valid_ratio + test_ratio != 100:
            raise HTTPException(
                status_code=400,
                detail="The sum of train, valid, and test ratios must equal 100."
            )

        # Create a temporary working directory
        tmpdir = tempfile.mkdtemp()

        # Save uploaded ZIP file
        input_zip_path = os.path.join(tmpdir, file.filename)
        with open(input_zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Temporary working directories
        extract_dir = os.path.join(tmpdir, "extracted")
        split_dir = os.path.join(tmpdir, "split")
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)

        # Extract the input zip
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Locate images and labels directories
        images_dir = os.path.join(extract_dir, "images")
        labels_dir = os.path.join(extract_dir, "labels")

        if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
            raise HTTPException(
                status_code=400,
                detail="Zip file must contain both 'images' and 'labels' folders!"
            )

        # Collect all image filenames (base name without extension)
        image_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No image files found in the 'images' folder. Supported formats: .jpg, .jpeg, .png"
            )

        random.shuffle(image_files)
        total = len(image_files)

        train_count = int(total * train_ratio / 100)
        valid_count = int(total * valid_ratio / 100)

        train_set = image_files[:train_count]
        valid_set = image_files[train_count:train_count + valid_count]
        test_set = image_files[train_count + valid_count:]

        # Helper to copy image-label pairs
        def copy_files(file_list, dest_folder):
            img_dest = os.path.join(split_dir, dest_folder, "images")
            lbl_dest = os.path.join(split_dir, dest_folder, "labels")
            os.makedirs(img_dest, exist_ok=True)
            os.makedirs(lbl_dest, exist_ok=True)

            for name in file_list:
                # Copy image
                for ext in (".jpg", ".jpeg", ".png"):
                    img_path = os.path.join(images_dir, name + ext)
                    if os.path.exists(img_path):
                        shutil.copy(img_path, img_dest)
                        break
                # Copy label
                label_path = os.path.join(labels_dir, name + ".txt")
                if os.path.exists(label_path):
                    shutil.copy(label_path, lbl_dest)

        # Perform copying
        copy_files(train_set, "train")
        copy_files(valid_set, "valid")
        copy_files(test_set, "test")

        # Create output zip
        output_zip_path = os.path.join(tmpdir, "dataset_split.zip")
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(split_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    arcname = os.path.relpath(abs_path, split_dir)
                    zipf.write(abs_path, arcname)

        # Ensure the zip file exists and has content
        if not os.path.exists(output_zip_path):
            raise HTTPException(status_code=500, detail="Failed to create output ZIP file")
        
        zip_size = os.path.getsize(output_zip_path)
        if zip_size == 0:
            raise HTTPException(status_code=500, detail="Output ZIP file is empty.")

        # Read the zip file content into memory
        with open(output_zip_path, 'rb') as f:
            zip_content = f.read()

        # Clean up temporary data
        shutil.rmtree(tmpdir)

        # Return ZIP file as streaming response
        def generate():
            yield zip_content

        response = StreamingResponse(
            generate(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="dataset_split.zip"',
                "Content-Length": str(zip_size),
                "X-Train-Count": str(len(train_set)),
                "X-Valid-Count": str(len(valid_set)),
                "X-Test-Count": str(len(test_set)),
                "X-Total-Count": str(total),
            }
        )
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video_trim/")
async def video_trim(file: UploadFile = File(...), start_time: str = Form(...), end_time: str = Form(...)):
    """
    Video trim: trims the video between start_time and end_time
    file: video file
    start_time: start time in seconds (numeric) or timestamp format (e.g., "00:00:10" or "00:01:23")
    end_time: end time in seconds (numeric) or timestamp format (e.g., "00:00:25" or "00:02:45")
    """
    clip = None
    trimmed = None
    
    try:
        # Convert timestamp string (like "00:01:23") to seconds if needed
        def to_seconds(t):
            if isinstance(t, str):
                # Try to parse as timestamp format first
                if ":" in t:
                    parts = [float(x) for x in t.split(":")]
                    if len(parts) == 3:
                        return parts[0]*3600 + parts[1]*60 + parts[2]
                    elif len(parts) == 2:
                        return parts[0]*60 + parts[1]
                # Otherwise try to parse as numeric string
                return float(t)
            return float(t)

        start = to_seconds(start_time)
        end = to_seconds(end_time)

        # Validate times
        if start < 0:
            raise HTTPException(status_code=400, detail="Start time must be non-negative.")
        if end <= start:
            raise HTTPException(status_code=400, detail="End time must be greater than start time.")

        # Create a temporary working directory
        tmpdir = tempfile.mkdtemp()

        # Save uploaded video file
        input_video_path = os.path.join(tmpdir, file.filename)
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load the video
        clip = VideoFileClip(input_video_path)
        original_duration = clip.duration

        # Validate end time doesn't exceed video duration
        if end > original_duration:
            raise HTTPException(
                status_code=400,
                detail=f"End time ({end}s) exceeds video duration ({original_duration:.2f}s)."
            )

        # Trim video (using slicing notation which works in all MoviePy versions)
        # For MoviePy 2.1.2+, subclip was renamed to subclipped, but slicing works universally
        try:
            # Try slicing notation first (works in newer versions)
            trimmed = clip[start:end]
        except (AttributeError, TypeError):
            # Fallback to subclipped for MoviePy 2.1.2+
            try:
                trimmed = clip.subclipped(start, end)
            except AttributeError:
                # Fallback to old subclip method for older MoviePy versions
                trimmed = clip.subclip(start, end)
        trimmed_duration = end - start
        
        # Determine output file extension (use .mp4 by default)
        file_ext = os.path.splitext(file.filename)[1] or ".mp4"
        output_video_path = os.path.join(tmpdir, f"trimmed_video{file_ext}")
        
        # Save result
        try:
            # Try with verbose parameter (older MoviePy versions)
            trimmed.write_videofile(
                output_video_path,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None
            )
        except TypeError:
            # Fallback without verbose parameter (newer MoviePy versions)
            trimmed.write_videofile(
                output_video_path,
                codec="libx264",
                audio_codec="aac"
            )

        # Ensure the output file exists and has content
        if not os.path.exists(output_video_path):
            raise HTTPException(status_code=500, detail="Failed to create trimmed video file")
        
        video_size = os.path.getsize(output_video_path)
        if video_size == 0:
            raise HTTPException(status_code=500, detail="Output video file is empty.")

        # Read the video file content into memory
        with open(output_video_path, 'rb') as f:
            video_content = f.read()

        # Clean up clips before removing temp directory
        if trimmed is not None:
            trimmed.close()
        if clip is not None:
            clip.close()

        # Clean up temporary data
        shutil.rmtree(tmpdir)

        # Determine content type based on file extension
        content_type_map = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
        }
        content_type = content_type_map.get(file_ext.lower(), 'video/mp4')

        # Return video file as streaming response
        def generate():
            yield video_content

        response = StreamingResponse(
            generate(),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="trimmed_video{file_ext}"',
                "Content-Length": str(video_size),
                "X-Original-Duration": str(original_duration),
                "X-Trimmed-Duration": str(trimmed_duration),
            }
        )
        
        return response

    except HTTPException:
        # Clean up clips if they were created
        if trimmed is not None:
            trimmed.close()
        if clip is not None:
            clip.close()
        raise
    except Exception as e:
        # Clean up clips if they were created
        if trimmed is not None:
            trimmed.close()
        if clip is not None:
            clip.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/frame_splitting/")
async def frame_splitting(file: UploadFile = File(...), frame_gap: int = Form(...), frame_name: str = Form(None)):
    """
    Frame splitting: extracts frames from a video file
    file: video file (MP4 format)
    frame_gap: extract every Nth frame (1 = every frame, 2 = every 2nd frame, etc.)
    frame_name: optional prefix name for the frames (default: "frame")
    """
    cap = None
    
    try:
        # Validate frame_gap
        if frame_gap < 1:
            raise HTTPException(status_code=400, detail="Frame gap must be at least 1.")
        
        # Create a temporary working directory
        tmpdir = tempfile.mkdtemp()
        
        # Save uploaded video file
        input_video_path = os.path.join(tmpdir, file.filename)
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Open the video
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Could not open video file: {file.filename}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output folder for frames
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Determine frame name prefix
        frame_prefix = frame_name if frame_name else "frame"
        
        # Extract frames
        count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every Nth frame
            if count % frame_gap == 0:
                filename = os.path.join(frames_dir, f"{frame_prefix}_{saved_count+1:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
            
            count += 1
        
        cap.release()
        
        if saved_count == 0:
            raise HTTPException(status_code=400, detail="No frames were extracted from the video.")
        
        # Create output zip file
        output_zip_path = os.path.join(tmpdir, "frames.zip")
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(frames_dir):
                for frame_file in files:
                    abs_path = os.path.join(root, frame_file)
                    arcname = os.path.relpath(abs_path, frames_dir)
                    zipf.write(abs_path, arcname)
        
        # Ensure the zip file exists and has content
        if not os.path.exists(output_zip_path):
            raise HTTPException(status_code=500, detail="Failed to create frames ZIP file")
        
        zip_size = os.path.getsize(output_zip_path)
        if zip_size == 0:
            raise HTTPException(status_code=500, detail="Frames ZIP file is empty.")
        
        # Read the zip file content into memory
        with open(output_zip_path, 'rb') as f:
            zip_content = f.read()
        
        # Clean up temporary data
        shutil.rmtree(tmpdir)
        
        # Return ZIP file as streaming response
        def generate():
            yield zip_content
        
        response = StreamingResponse(
            generate(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="frames.zip"',
                "Content-Length": str(zip_size),
                "X-Total-Frames": str(frame_count),
                "X-Extracted-Frames": str(saved_count),
                "X-Frame-Gap": str(frame_gap),
                "X-FPS": str(fps) if fps else "0",
            }
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        if cap is not None:
            cap.release()
        raise HTTPException(status_code=500, detail=str(e))