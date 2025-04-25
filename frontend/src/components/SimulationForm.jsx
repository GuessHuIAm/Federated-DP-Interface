import { useState } from 'react';
import { Button, Slider, Select, MenuItem, FormControl, InputLabel, Typography } from '@mui/material';
import SimulationStream from './SimulationStream';

function SimulationForm() {
  const [epsilon, setEpsilon] = useState(1.0);
  const [clip, setClip] = useState(1.0);
  const [numClients, setNumClients] = useState(5);
  const [mechanism, setMechanism] = useState('Gaussian');
  const [rounds, setRounds] = useState(5);
  const [start, setStart] = useState(false);

  const handleStart = () => {
    setStart(true);
  };

  if (start) {
    return (
      <SimulationStream
        epsilon={epsilon}
        clip={clip}
        numClients={numClients}
        mechanism={mechanism}
        rounds={rounds}
        onBack={() => setStart(false)}
      />
    );
  }

  return (
    <div style={{ padding: '20px' }}>
      <Typography variant="h4" gutterBottom>
        Differential Privacy FL Explorer
      </Typography>

      <Typography gutterBottom>Privacy parameter ε</Typography>
      <Slider value={epsilon} min={0.1} max={10.0} step={0.1} onChange={(e, val) => setEpsilon(val)} valueLabelDisplay="auto"/>

      <Typography gutterBottom>Clipping Norm</Typography>
      <Slider value={clip} min={0.1} max={5.0} step={0.1} onChange={(e, val) => setClip(val)} valueLabelDisplay="auto"/>

      <Typography gutterBottom>Number of Clients</Typography>
      <Slider value={numClients} min={1} max={20} step={1} onChange={(e, val) => setNumClients(val)} valueLabelDisplay="auto"/>

      <Typography gutterBottom>Rounds</Typography>
      <Slider value={rounds} min={1} max={100} step={1} onChange={(e, val) => setRounds(val)} valueLabelDisplay="auto"/>

      <FormControl fullWidth style={{ marginTop: '20px' }}>
        <InputLabel id="mechanism-label">DP Mechanism</InputLabel>
        <Select
          labelId="mechanism-label"
          value={mechanism}
          label="DP Mechanism"
          onChange={(e) => setMechanism(e.target.value)}
        >
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
        Start Training
      </Button>
    </div>
  );
}

export default SimulationForm;