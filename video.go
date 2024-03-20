package supervisiongo

import (
	"errors"
	"image"

	"gocv.io/x/gocv"
)
type VideoInfo struct {
    Width           int
    Height          int
    fps             int
    TotalFrame      int
    ResizeWidth     int
    ResizeHeight    int
    Resize          float32 
}

func NewVideoInfoFromPath (sourcePath string, resize float32) (*VideoInfo, error){
    video, err := gocv.OpenVideoCapture(sourcePath)
    if err != nil {
        return nil, err
    }
    if !video.IsOpened() {
        return nil, errors.New("cannot open video capture")
    }
    width := int(video.Get(gocv.VideoCaptureFrameWidth))
    height := int(video.Get(gocv.VideoCaptureFrameHeight))
    fps := int(video.Get(gocv.VideoCaptureFPS))
    totalFrame := int(video.Get(gocv.VideoCaptureFrameCount))

    err = video.Close()
    if err != nil {
        return nil, err
    }

    return &VideoInfo{
        Width: width,
        Height: height,
        fps: fps,
        TotalFrame: totalFrame,
        Resize: resize,
        ResizeWidth: int(float32(width) * resize),
        ResizeHeight: int(float32(height) * resize),
    }, nil
}

type VideoSink struct {
    VideoWriter *gocv.VideoWriter
    VideoInfo   *VideoInfo
    Codec       string
    TargetPath  string
}

func NewVideoSink(targetPath string, videoInfo *VideoInfo, codec string) (*VideoSink, error) {
    videoWriter, err := gocv.VideoWriterFile(targetPath, codec, float64(videoInfo.fps), videoInfo.ResizeWidth, videoInfo.ResizeHeight, true)
    if err != nil {
        return nil, err
    }

    return &VideoSink{
        VideoWriter: videoWriter,
        VideoInfo: videoInfo,
        Codec: codec,
        TargetPath: targetPath,
    }, nil
}

func (v *VideoSink) WriteFrame(frame gocv.Mat) error {
    err := v.VideoWriter.Write(frame)
    if err != nil {
        return err
    }
    return nil
}

func (v *VideoSink) Destroy() error{
    err := v.VideoWriter.Close()

    if err != nil {
        return err
    }
    return nil
}


func VideoFrameGenerator (yield func(gocv.Mat) bool, sourcePath string) {
    video, err := gocv.OpenVideoCapture(sourcePath)
    if err != nil {
        return
    }
    defer video.Close()

    frame := gocv.NewMat()
    defer frame.Close()

    for {
        if ok := video.Read(&frame); !ok {
            break
        }
        if frame.Empty() {
            continue
        }

        if !yield(frame) {
            return 
        }
    }
}

func ProcessVideo(sourcePath, targetPath, codec string, resize float32,callback func(frame gocv.Mat) error) error {
    sourceVideoInfo, err := NewVideoInfoFromPath(sourcePath, resize)
    if err != nil {
        return err 
    }

    videoSink, err := NewVideoSink(targetPath, sourceVideoInfo, codec)
    defer videoSink.Destroy()

    if err != nil {
        return err
    }


    yield := func(frame gocv.Mat) bool {
        gocv.Resize(frame, &frame, image.Point{X: sourceVideoInfo.ResizeWidth, Y: sourceVideoInfo.ResizeHeight}, 0, 0, gocv.InterpolationLinear)
        err := callback(frame) 
        if err != nil {
            return false
        }
        err = videoSink.WriteFrame(frame)
        if err != nil {
            return false
        }
        return true
    }

    VideoFrameGenerator(yield, sourcePath)
    return nil
}

