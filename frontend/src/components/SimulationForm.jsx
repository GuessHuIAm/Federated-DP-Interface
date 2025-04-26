import { useState } from 'react';
import { Button, Slider, Select, MenuItem, FormControl, InputLabel, Typography, Tooltip } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import SimulationStream from './SimulationStream';
import TrainingInfoGraphic from './TrainingInfoGraphic';

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
        Differential Privacy Federated Learning Explorer
      </Typography>

      <TrainingInfoGraphic />

      {/* Privacy parameter ε */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '20px' }}>
        <Typography gutterBottom>Privacy parameter ε</Typography>
        <Tooltip
          title="Controls how much noise is added. Lower ε = stronger privacy but may hurt model accuracy."
          arrow
          placement="right"
          componentsProps={{
            tooltip: { sx: { fontSize: '18px' } }
          }}
        >
          <InfoOutlinedIcon fontSize="small" />
        </Tooltip>
      </div>
      <Slider value={epsilon} min={0.1} max={10.0} step={0.1} onChange={(e, val) => setEpsilon(val)} valueLabelDisplay="auto" />

      {/* Clipping Norm */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '24px' }}>
        <Typography gutterBottom>Clipping Norm</Typography>
        <Tooltip
          title="Limits the size of each client's model update. Smaller norms improve privacy protection but may slow learning if updates are heavily clipped."
          arrow
          placement="right"
          componentsProps={{
            tooltip: { sx: { fontSize: '18px' } }
          }}
        >
          <InfoOutlinedIcon fontSize="small" />
        </Tooltip>
      </div>
      <Slider value={clip} min={0.1} max={5.0} step={0.1} onChange={(e, val) => setClip(val)} valueLabelDisplay="auto" />

      {/* Number of Clients */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '24px' }}>
        <Typography gutterBottom>Number of Clients</Typography>
        <Tooltip
          title="The number of clients participating in each round. More clients lead to more stable updates but require more communication."
          arrow
          placement="right"
          componentsProps={{
            tooltip: { sx: { fontSize: '18px' } }
          }}
        >
          <InfoOutlinedIcon fontSize="small" />
        </Tooltip>
      </div>
      <Slider value={numClients} min={1} max={20} step={1} onChange={(e, val) => setNumClients(val)} valueLabelDisplay="auto" />

      {/* Rounds */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '24px' }}>
        <Typography gutterBottom>Rounds</Typography>
        <Tooltip
          title="Total number of communication rounds. More rounds help the model converge but can increase total privacy loss over time."
          arrow
          placement="right"
          componentsProps={{
            tooltip: { sx: { fontSize: '18px' } }
          }}
        >
          <InfoOutlinedIcon fontSize="small" />
        </Tooltip>
      </div>
      <Slider value={rounds} min={1} max={100} step={1} onChange={(e, val) => setRounds(val)} valueLabelDisplay="auto" />

      {/* DP Mechanism */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '30px' }}>
        <InputLabel id="mechanism-label">DP Mechanism</InputLabel>
          <Tooltip
            title={
              "Chooses the type of noise added to protect updates:\n\n" +
              "• Gaussian noise: Adds normally distributed noise; commonly used for scalable and flexible privacy guarantees.\n\n" +
              "• Laplace noise: Adds noise from a Laplace distribution; can provide tighter privacy for small datasets but is more sensitive to extreme updates."
            }
            arrow
            placement="right"
            componentsProps={{
              tooltip: { sx: { fontSize: '18px', whiteSpace: 'pre-line' } }
            }}
          >
            <InfoOutlinedIcon fontSize="small" />
          </Tooltip>
      </div>
      <FormControl fullWidth style={{ marginTop: '8px' }}>
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
