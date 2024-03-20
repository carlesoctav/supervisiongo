package supervisiongo

import (
	"fmt"
	"sort"

	"image"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

var DEFAULTCLASSES = []string{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}


type YoloOutput struct {
    Xyxy [][]int
    Classes []int
    Confidences []float32
    TrackerIds []int
    Labels []string
    Data map[string]interface{}
}

type BoundingBox struct {
    label      string
    confidence float32
    classId  int
    x1, y1, x2, y2 float32
}

func (b *BoundingBox) String() string {
    return fmt.Sprintf("Object %s (confidence %f): (%f, %f), (%f, %f)",
        b.label, b.confidence, b.x1, b.y1, b.x2, b.y2)
}

func (b *BoundingBox) ToReact() image.Rectangle {
    return image.Rect(int(b.x1), int(b.y1), int(b.x2), int(b.y2)).Canon()
}

func (b *BoundingBox) RectArea() int {
    size := b.ToReact().Size()
    return size.X * size.Y
}

func (b *BoundingBox) Intersection(other *BoundingBox) float32 {
    r1 := b.ToReact()
    r2 := other.ToReact()
    intersected := r1.Intersect(r2).Canon().Size()
    return float32(intersected.X * intersected.Y)
}

func (b *BoundingBox) Union(other *BoundingBox) float32 {
    intersectArea := b.Intersection(other)
    totalArea := float32(b.RectArea() + other.RectArea())
    return totalArea - intersectArea
}

func (b *BoundingBox) Iou(other *BoundingBox) float32 {
    return b.Intersection(other) / b.Union(other)
}


type ModelSession struct {
    Session *ort.AdvancedSession
    Input   *ort.Tensor[float32]
    Output  *ort.Tensor[float32]
}

func (m *ModelSession) Destroy() {
    m.Session.Destroy()
    m.Input.Destroy()
    m.Output.Destroy()
}


type Yolo struct {
    ModelPath string
    ModelSession *ModelSession
    Classes []string
    TotalClasses int
    IouThreshold float32
    ConfidenceThreshold float32
}

func (y *Yolo) Destroy() {
    y.ModelSession.Destroy()
}

type YoloOptions func(*Yolo) error

func NewYolo( opts ...YoloOptions) (*Yolo, error) {
    yolo := &Yolo{
        ModelPath: "./yolov8n.onnx",
        ModelSession: nil,
        Classes: DEFAULTCLASSES,
        TotalClasses: 80,
        IouThreshold: 0.7,
        ConfidenceThreshold: 0.5,
    }

    for _, opt := range opts {
        err := opt(yolo)
        if err != nil {
            return nil, err
        }
    }
    err := yolo.initSession()
    if err != nil {
        return nil, err
    }

    return yolo, nil
}

func (y *Yolo) initSession() error{
    ort.SetSharedLibraryPath("./onnxruntime-linux-x64-1.17.1/lib/libonnxruntime.so")
    err := ort.InitializeEnvironment()
    if err != nil {
        return fmt.Errorf("Error initializing ORT environment: %w", err)
    }

    inputShape := ort.NewShape(1, 3, 640, 640)
    inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
    if err != nil {
        return fmt.Errorf("Error creating input tensor: %w", err)
    }
    outputShape := ort.NewShape(1, 84, 8400)
    outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
    if err != nil {
        inputTensor.Destroy()
        return  fmt.Errorf("Error creating output tensor: %w", err)
    }
    options, err := ort.NewSessionOptions()
    if err != nil {
        inputTensor.Destroy()
        outputTensor.Destroy()
        return fmt.Errorf("Error creating ORT session options: %w", err)
    }
    defer options.Destroy()
    session, err := ort.NewAdvancedSession(y.ModelPath,
        []string{"images"}, []string{"output0"},
        []ort.ArbitraryTensor{inputTensor},
        []ort.ArbitraryTensor{outputTensor},
        options)

    if err != nil {
        inputTensor.Destroy()
        outputTensor.Destroy()
        return fmt.Errorf("Error creating ORT session: %w", err)
    }

    y.ModelSession = &ModelSession{
        Session: session,
        Input:   inputTensor,
        Output:  outputTensor,
    }

    return nil
}

func (y *Yolo) prepareInput(img image.Image) error {
    data := y.ModelSession.Input.GetData()
    channelSize := 640 * 640
    if len(data) < (channelSize * 3) {
        return fmt.Errorf("Destination tensor only holds %d floats, needs "+
            "%d (make sure it's the right shape!)", len(data), channelSize*3)
    }
    redChannel := data[0:channelSize]
    greenChannel := data[channelSize : channelSize*2]
    blueChannel := data[channelSize*2 : channelSize*3]

    img = resize.Resize(640, 640, img, resize.Lanczos3)
    i := 0
    for y := 0; y < 640; y++ {
        for x := 0; x < 640; x++ {
            r, g, b, _ := img.At(x, y).RGBA()
            redChannel[i] = float32(r>>8) / 255.0
            greenChannel[i] = float32(g>>8) / 255.0
            blueChannel[i] = float32(b>>8) / 255.0
            i++
        }
    }
    return nil
}

func (y *Yolo) Predict(mat gocv.Mat) ([]BoundingBox, error) {
    img, err := mat.ToImage()
    originalWidth, originalHeight := img.Bounds().Dx(), img.Bounds().Dy()
    if err != nil {
        return nil, err
    }
    err = y.prepareInput(img)
    if err != nil {
        return nil, err
    }

    err = y.ModelSession.Session.Run()

    if err != nil {
        return nil, err
    }

    yoloOutput := y.processOutput(y.ModelSession.Output.GetData(), originalWidth, originalHeight)

    return yoloOutput, nil
}

func (y *Yolo) processOutput(output []float32, originalWidth,
        originalHeight int) ([]BoundingBox) {
    boundingBoxes := make([]BoundingBox, 0, 8400)

    var classID int
    var probability float32

    for idx := 0; idx < 8400; idx++ {
        // Iterate through 80 classes and find the class with the highest probability
        probability = -1e9
        for col := 0; col < y.TotalClasses; col++ {
            currentProb := output[8400*(col+4)+idx]
            if currentProb > probability {
                probability = currentProb
                classID = col
            }
        }

        if probability < y.ConfidenceThreshold { 
            continue
        }

        xc, yc := output[idx], output[8400+idx]
        w, h := output[2*8400+idx], output[3*8400+idx]
        x1 := (xc - w/2) / 640 * float32(originalWidth)
        y1 := (yc - h/2) / 640 * float32(originalHeight)
        x2 := (xc + w/2) / 640 * float32(originalWidth)
        y2 := (yc + h/2) / 640 * float32(originalHeight)

        boundingBoxes = append(boundingBoxes, BoundingBox{
            label:      y.Classes[classID],
            classId:   classID,
            confidence: probability,
            x1:         x1,
            y1:         y1,
            x2:         x2,
            y2:         y2,
        })
    }

    // Sort the bounding boxes by probability
    sort.Slice(boundingBoxes, func(i, j int) bool {
        return boundingBoxes[i].confidence < boundingBoxes[j].confidence
    })

    mergedResults := make([]BoundingBox, 0, len(boundingBoxes))


    // Iterate through sorted bounding boxes, removing overlaps
    for _, candidateBox := range boundingBoxes {
        overlapsExistingBox := false
        for _, existingBox := range mergedResults {
            if (&candidateBox).Iou(&existingBox) > y.IouThreshold {
                overlapsExistingBox = true
                break
            }
        }

        if !overlapsExistingBox {
            mergedResults = append(mergedResults, candidateBox)
        }
    }

    return mergedResults

}
