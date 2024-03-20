package main

import (
	"fmt"
	"image/color"
	"os"

	"github.com/supervisiongo"
	"gocv.io/x/gocv"
)


func run() int {
    yolo, err := supervisiongo.NewYolo()
    if err != nil {
        fmt.Errorf("Error creating Yolo: %w", err)
        return 1
    }
    window := gocv.NewWindow("apple tree")
    callback := func(mat gocv.Mat) ( error){
        img, err:= mat.ToImage()

        if err != nil {
            return err
        }
        boxes, err := yolo.Predict(img)

        for _, box := range boxes {
            gocv.Rectangle(&mat, box.ToReact(), color.RGBA{0, 255, 0, 0}, 2)
        }

        window.IMShow(mat)
        if window.WaitKey(1) >= 0 {
            return fmt.Errorf("window closed")
        }
        return nil
    }

   supervisiongo.ProcessVideo("./media/vehicle.mp4", "./media/vehicle_output.mp4", "mp4v", 0.2, callback)
    return 0
}

func main(){
    os.Exit(run())
}
