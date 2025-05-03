import React, { useState } from 'react';
import {
  Button, Slider, Select, MenuItem, FormControl,
  InputLabel, Typography, TextField, Tooltip
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import SweepResults from './SweepResults';
import TrainingInfoGraphic from './TrainingInfoGraphic';

function SimulationForm() {
  const [epsilon, setEpsilon] = useState(1.0);
  const [clip, setClip] = useState(1.0);
  const [numClients, setNumClients] = useState(5);
  const [mechanism, setMechanism] = useState('Gaussian');
  const [rounds, setRounds] = useState(5);

  const [sweepParam, setSweepParam] = useState('numClients');
  const [sweepValues, setSweepValues] = useState('2,4,6,8');
  const [start, setStart] = useState(false);
  const [sweepConfig, setSweepConfig] = useState(null);

  const handleStart = () => {
    const parsedValues = sweepValues.split(',').map(v => Number(v.trim()));
    setSweepConfig({ param: sweepParam, values: parsedValues });
    setStart(true);
  };

  const setDefault = (param) => {
    if (param === 'epsilon') setEpsilon(1.0);
    else if (param === 'clip') setClip(1.0);
    else if (param === 'numClients') setNumClients(5);
    else if (param === 'rounds') setRounds(10);
  };

  const sweepParamTooltips = {
    epsilon: "Controls how much noise is added. Lower ε = stronger privacy but may hurt model accuracy.",
    numClients: "The number of clients participating in each round. More clients lead to more stable updates but require more communication.",
    rounds: "Total number of communication rounds. More rounds help the model converge but can increase total privacy loss over time.",
  };
  

  if (start) {
    return (
      <SweepResults
        param={sweepConfig.param}
        values={sweepConfig.values}
        epsilon={epsilon}
        clip={clip}
        mechanism={mechanism}
        rounds={rounds}
        numClients={numClients}
        onBack={() => setStart(false)}
      />
    );
  }

  const ParameterControl = ({ label, tooltip, value, onChange, min, max, step, paramKey }) => (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Typography gutterBottom sx={{ mb: 0 }}>{label}</Typography>
          <Tooltip title={tooltip} arrow>
            <InfoOutlinedIcon fontSize="small" />
          </Tooltip>
        </div>
        <Button size="small" onClick={() => setDefault(paramKey)}>Set Default</Button>
      </div>
      <Slider value={value} min={min} max={max} step={step} onChange={(e, val) => onChange(val)} valueLabelDisplay="auto" />
    </>
  );

  return (
    <div style={{ padding: '20px' }}>
      <Typography variant="h4" gutterBottom>
        Differential Privacy Federated Learning Explorer
      </Typography>

      <TrainingInfoGraphic />

      <div style={{ marginTop: '24px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
      <Typography sx={{ fontSize: '14px', fontWeight: 500 }}>Parameter to Evaluate</Typography>
      <Tooltip
        title={sweepParamTooltips[sweepParam] || "Select which parameter to vary. The others will be held constant."}
        arrow
      >
        <InfoOutlinedIcon fontSize="small" />
      </Tooltip>
    </div>
    <FormControl fullWidth>
      <Select value={sweepParam} onChange={(e) => setSweepParam(e.target.value)}>
        <MenuItem value="numClients">Number of Clients</MenuItem>
        <MenuItem value="epsilon">Epsilon</MenuItem>
        <MenuItem value="rounds">Rounds</MenuItem>
      </Select>
    </FormControl>
  </div>


      {/* Parameter Sweep Controls
      <FormControl fullWidth style={{ marginTop: '24px' }}>
        <InputLabel>Parameter to Evaluate</InputLabel>
        <Select value={sweepParam} onChange={(e) => setSweepParam(e.target.value)}>
          <MenuItem value="numClients">Number of Clients</MenuItem>
          <MenuItem value="epsilon">Epsilon</MenuItem>
          <MenuItem value="rounds">Rounds</MenuItem>
        </Select>
      </FormControl> */}

      <TextField
        fullWidth
        style={{ marginTop: '12px' }}
        label="Values to Evaluate (comma-separated)"
        value={sweepValues}
        onChange={(e) => setSweepValues(e.target.value)}
      />

      <Typography variant="h6" sx={{ mt: 4 }}>Fixed Parameters</Typography>

      {sweepParam !== 'epsilon' && (
        <ParameterControl
          label="Privacy parameter ε"
          tooltip="Controls how much noise is added. Lower ε = stronger privacy but may hurt model accuracy."
          value={epsilon}
          onChange={setEpsilon}
          min={0.1}
          max={10.0}
          step={0.1}
          paramKey="epsilon"
        />
      )}

      {/* {sweepParam !== 'clip' && (
        <ParameterControl
          label="Clipping Norm"
          tooltip="Limits the size of each client's model update. Smaller norms improve privacy protection but may slow learning if updates are heavily clipped."
          value={clip}
          onChange={setClip}
          min={0.1}
          max={5.0}
          step={0.1}
          paramKey="clip"
        />
      )} */}

      {sweepParam !== 'numClients' && (
        <ParameterControl
          label="Number of Clients"
          tooltip="The number of clients participating in each round. More clients lead to more stable updates but require more communication."
          value={numClients}
          onChange={setNumClients}
          min={1}
          max={20}
          step={1}
          paramKey="numClients"
        />
      )}

      {sweepParam !== 'rounds' && (
        <ParameterControl
          label="Rounds"
          tooltip="Total number of communication rounds. More rounds help the model converge but can increase total privacy loss over time."
          value={rounds}
          onChange={setRounds}
          min={1}
          max={100}
          step={1}
          paramKey="rounds"
        />
      )}

      <FormControl fullWidth style={{ marginTop: '24px' }}>
        <InputLabel>DP Mechanism</InputLabel>
        <Select value={mechanism} onChange={(e) => setMechanism(e.target.value)}>
          <MenuItem value="Gaussian">Gaussian</MenuItem>
          <MenuItem value="Laplace">Laplace</MenuItem>
        </Select>
      </FormControl>

      <Button
        variant="contained"
        color="primary"
        style={{ marginTop: '30px' }}
        fullWidth
        onClick={handleStart}
      >
        Start Parameter Evaluation
      </Button>
    </div>
  );
}

export default SimulationForm;
