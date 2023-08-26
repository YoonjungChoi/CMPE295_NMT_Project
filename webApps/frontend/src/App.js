import React from 'react';
import './App.css';

import Home from './components/Home';


function App() {
  return (
    <div className="AppHEAD">
      <div> 
        <h1>
          Neural Machine Translation
        </h1>
        <h2>
          Between Korean and English
        </h2>
      </div>
      <div className="AppSection"/>
      <div>
        <Home/>
      </div>
    </div>
  );
}

export default App;
