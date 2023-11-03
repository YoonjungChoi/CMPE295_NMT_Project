import React from 'react';
import axios from 'axios';
import './Home.css';

import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';

//API_URL = "http://localhost:8080/doTranslation/";

class Home extends React.Component{

  constructor(props) {
    super(props);
    this.state = {
        srcEn: "",
        tgtKo: "",
        srcKo: "",
        tgtEn: ""
    };
  }
  
  handleChangeSrcEn = (e) => {
    //console.log("LOG-handleChangeSrcEn()", e.target.value);
    this.setState({
        srcEn: e.target.value,
    })
  }

  handleChangeSrcKo = (e) => {
    //console.log("LOG-handleChangeSrcKo()", e.target.value);
    this.setState({
        srcKo: e.target.value,
    })
  }

  doTranslationEnKo = (e) => {
    e.preventDefault();
    console.log("LOG-doTranslationEnKo", this.state.srcEn);
    this.setState({
      tgtKo: "..."
    });
    this.request = { src: "en", tgt: "ko", srcText: this.state.srcEn};
    axios.post(
      "http://127.0.0.1:8000/doTranslation/",
       this.request
      ).then(response=> {
        console.log("LOG-SUCCESS", response.data);
        this.setState({tgtKo: response.data.tgtText});
      }).catch(error=> {
        console.log("LOG-ERROR", error.message);
      });
  }

  doTranslationKoEn = (e) => {
    e.preventDefault();
    console.log("LOG-doTranslationKoEn");
    this.setState({
      tgtEn: "..."
    });
    this.request = { src: "ko", tgt: "en", srcText:this.state.srcKo };
    axios.post(
        "http://127.0.0.1:8000/doTranslation/",
        this.request
      ).then(response=> {
        console.log("LOG-SUCCESS", response.data);
        this.setState({tgtEn: response.data.tgtText});
      }).catch(error=> {
        console.log("LOG-ERROR", error.message);
      });
  }


  render() {
  return( 
    <div className="HomeMain">
      <div className = "HomeSubOneTitle">
        <h2>English To korean</h2>            
      </div>
      <div  className = "HomeSubOne">      
        <TextField
              id="div-inner"
              label="English"
              variant="outlined"
              multiline
              rowsMax="4"
              value={this.state.srcEn}
              onChange={this.handleChangeSrcEn}
          />
          <Button id="div-inner" variant="contained" onClick={this.doTranslationEnKo}>Do</Button> 
          <TextField
              id="div-inner"
              value={this.state.tgtKo}
              multiline
          />
      </div>

      <div className="HomeSection"/>
          
      <div className = "HomeSubTwoTitle">
        <h2>Korean To English</h2>            
      </div>

      <div className = "HomeSubTwo">
        <TextField
            id="div-inner"
            label="Korean"
            variant="outlined"
            multiline
            rowsMax="4"
            value={this.state.srcKo}
            onChange={this.handleChangeSrcKo}
        />
        <Button id="div-inner" variant="contained" onClick={this.doTranslationKoEn}>Do</Button> 
        <TextField
            id="div-inner"
            value={this.state.tgtEn}
            multiline
        />
      </div>

    </div>
  )}
}
export default Home;