import React, { useEffect, useRef, useState } from "react";

import Home from "./containers/Home";

import "./App.css";

function App() {
  return (
    <div className="App" style={{ height: "100%" }}>
      <style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100;0,400;1,200&display=swap');
</style>
      <Home />
    </div>
  );
}

export default App;
