import React from "react";

import styles from "./image_viewer.module.scss";
import Lottie from "lottie-react";
import Sun from "../../assets/animation_lnhjmtfp.json"
import MidSun from "../../assets/mid_sun.json"
import RedSun from "../../assets/red_sun.json"

interface Props {
  kValue: number | null;
}

const suns = [Sun, MidSun, RedSun]

const ImageViewer: React.FC<Props> = ({kValue}) => {
  return (
    <div
      className={styles.image__container
      }
    >
    <Lottie animationData={Sun} loop={true} />
    </div>
  );
};

export default ImageViewer;
