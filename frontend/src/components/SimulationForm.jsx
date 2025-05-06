import React, { useState } from 'react';
import {
  Button, Slider, FormControl,
  InputLabel, Typography, TextField, Tooltip, Select, MenuItem
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import SweepResults from './SweepResults';
import TrainingInfoGraphic from './TrainingInfoGraphic';

function SimulationForm() {
  const [epsilon, setEpsilon] = useState(1.0);
  const [numClients, setNumClients] = useState(5);
  const [rounds, setRounds] = useState(5);

  const [sweepParam, setSweepParam] = useState('numClients');
  const [sweepValues, setSweepValues] = useState('2,4,6,8');
  const [start, setStart] = useState(false);
  const [sweepConfig, setSweepConfig] = useState(null);

  const handleStart = () => {
    const parsedValues = sweepValues
      .split(',')
      .map(v => Number(v.trim()))
      .filter(v => {
        if (sweepParam === 'epsilon') return !isNaN(v) && v > 0;
        return Number.isInteger(v) && v > 0;
      });

    if (parsedValues.length === 0) {
      alert("Invalid values. Please correct the input.");
      return;
    }

    const fixedParams = {
      epsilon,
      numClients,
      rounds,
    };

    delete fixedParams[sweepParam];

    setSweepConfig({
      param: sweepParam,
      values: parsedValues,
      fixed: fixedParams,
    });

    setStart(true);
  };

  const setDefault = (param) => {
    if (param === 'epsilon') setEpsilon(1.0);
    else if (param === 'numClients') setNumClients(5);
    else if (param === 'rounds') setRounds(10);
  };

  const isValidSweepValues = () => {
    const parsed = sweepValues.split(',').map(v => Number(v.trim()));
    if (sweepParam === 'epsilon') {
      return parsed.every(v => !isNaN(v) && v > 0);
    }
    return parsed.every(v => Number.isInteger(v) && v > 0);
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
        fixed={sweepConfig.fixed}
        epsilon={epsilon}
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
      <Slider value={value} min={min} max={max} step={step} onChange={(e, val) => onChange(val)} valueLabelDisplay="on" />
    </>
  );

  return (
    <div style={{ padding: '20px' }}>
      <Typography variant="h4" gutterBottom>
        Differential Privacy Federated Learning Explorer
      </Typography>

      <TrainingInfoGraphic />

      <div style={{ marginTop: '24px' }}>
        <Typography sx={{ fontSize: '14px', fontWeight: 500 }}>Parameter to Evaluate</Typography>
        <FormControl fullWidth>
          <Select value={sweepParam} onChange={(e) => setSweepParam(e.target.value)}>
            <MenuItem value="numClients">Number of Clients</MenuItem>
            <MenuItem value="epsilon">Epsilon</MenuItem>
            <MenuItem value="rounds">Rounds</MenuItem>
          </Select>
        </FormControl>
      </div>

      <TextField
        fullWidth
        style={{ marginTop: '12px' }}
        label="Values to Evaluate (comma-separated)"
        value={sweepValues}
        onChange={(e) => setSweepValues(e.target.value)}
        error={!isValidSweepValues()}
        helperText={
          !isValidSweepValues()
            ? sweepParam === 'epsilon'
              ? "Enter only positive numbers (e.g., 0.5, 1.0, 2)"
              : "Enter only positive integers (e.g., 2, 4, 6)"
            : " "
        }
      />

      <Typography variant="h6" sx={{ mt: 4 }}>Fixed Parameters</Typography>

      {sweepParam !== 'epsilon' && (
        <ParameterControl
          label="Privacy parameter ε"
          tooltip={sweepParamTooltips.epsilon}
          value={epsilon}
          onChange={setEpsilon}
          min={0.1}
          max={10.0}
          step={0.1}
          paramKey="epsilon"
        />
      )}

      {sweepParam !== 'numClients' && (
        <ParameterControl
          label="Number of Clients"
          tooltip={sweepParamTooltips.numClients}
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
          tooltip={sweepParamTooltips.rounds}
          value={rounds}
          onChange={setRounds}
          min={1}
          max={100}
          step={1}
          paramKey="rounds"
        />
      )}

      <Button
        variant="contained"
        color="primary"
        style={{ marginTop: '30px' }}
        fullWidth
        onClick={handleStart}
        disabled={!isValidSweepValues()}
      >
        Start Parameter Evaluation
      </Button>
    </div>
  );
}

export default SimulationForm;
