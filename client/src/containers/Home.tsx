import React, { useRef, useEffect, useState } from "react";
import ImageViewer from "../components/image_viewer/image_viewer";
import styles from "./home.module.scss";
import { Console } from "console";
import axios from "axios"
import { createEnumDeclaration } from "typescript";

interface DataFile {
  textFile: string | null;
  filename: string | null;
}

const colors = ["#4cc73c", "#4cc73c", "#66c73c", "#66c73c", "#66c73c", "#8dc73c", "#c2c73c", "#c79d3c", "#c44e2d", "#c42d2d"]


const Home = () => {

  const [kValue, setKValue] = useState(-1)
  const [kValue3, setKValue3] = useState(-1)
  const [kTime, setKTime] = useState('')

  const requestData = async () => {
    try {
    let response = await axios.get("http://127.0.0.1:5000/predict/now")
    setKValue(response.data.prediction[0].kp)
    setKValue3(response.data.prediction[1].kp)
    console.log(response.data.timestamp)
    setKTime(response.data.timestamp)
    } catch (err) {
      console.error(err)
    }
  }

  //setInterval(requestData, 2000)

  useEffect(() => {
    const interval = setInterval(async () => {
      await requestData();
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  //setInterval(requestData, 2000)

  return (
    <div className={styles.center}>
      <div className={styles.container}>
        <ImageViewer kValue={kValue}/>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16, width: '340px'}}>
          <div style={{ textAlign: "center" }}>
            {kTime}
          </div>
          <div style={{display: "flex", flexDirection: "row"}}>
            <span>
              <div style={{textAlign: "center", width: '260px'}}>Current k-index:</div>
              <div className={styles.k_display} style={{color: colors[Math.trunc(kValue)]}}>{kValue.toFixed(2)}</div>
            </span>
            <span>
              <div style={{textAlign: "center", width: '260px'}}>~3 hours k-index estimation:</div>
              <div className={styles.k_display} style={{color: colors[Math.trunc(kValue3)]}}>{kValue3.toFixed(2)}</div>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
