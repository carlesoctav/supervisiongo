package main

import (
	"fmt"
	"image/color"

	"github.com/supervisiongo"
	"gocv.io/x/gocv"
)
func main(){

	yolo, err := supervisiongo.NewYolo()
	defer yolo.Destroy()
	if err != nil {
		fmt.Println(err)
		return 
	}

	mat := gocv.NewMat()
	defer mat.Close()
	
	window := gocv.NewWindow("Yolo")
	defer window.Close()

	webcam, err := gocv.OpenVideoCapture("./media/vehicle.mp4")

	for {
		if ok:= webcam.Read(&mat); !ok {
			fmt.Printf("cannot read device")
			return 
		}
		if mat.Empty() {
			continue
		}

		img ,err := mat.ToImage()
		if err != nil {
			return 
		}
		boxes, err := yolo.Predict(img)

		if err != nil {
			fmt.Println(err)
			return 
		}

		for  _, box := range boxes {
			gocv.Rectangle(&mat, box.ToReact(), color.RGBA{0, 255, 0, 0}, 2)
		}

		window.IMShow(mat)
		if window.WaitKey(1) >= 0 {
			break
		}
	}

}

