import React, { useState } from 'react';
import BaseButtonModal from './BaseButtonModal';
import { Typography } from '@mui/material';
import Slider from 'react-slick';
import SchoolIcon from '@mui/icons-material/School';
import ContentCutIcon from '@mui/icons-material/ContentCut';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import SendIcon from '@mui/icons-material/Send';
import PublicIcon from '@mui/icons-material/Public';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import { PrevArrow, NextArrow } from './Arrows';

const steps = [
  {
    title: 'Model Starts Empty on Your Device',
    description: 'Your phone begins with a small, untrained model ready to learn from your local data — without sending anything away.',
    Icon: SchoolIcon,
  },
  {
    title: 'Train Locally Using Your Private Data',
    description: 'Your device teaches the model using your health data, making it smarter — but keeping the data private.',
    Icon: SchoolIcon,
  },
  {
    title: 'Limit How Much Each Update Can Say',
    description: 'We carefully clip your model updates, so no single piece of data can have too much influence.',
    Icon: ContentCutIcon,
  },
  {
    title: 'Add Protective Random Noise',
    description: 'Before sending anything, random noise is mixed into your updates to protect your privacy even more.',
    Icon: VolumeUpIcon,
  },
  {
    title: 'Send Only Protected Updates',
    description: 'Only the noisy, privacy-protected model updates (never your raw data) are sent securely to the server.',
    Icon: SendIcon,
  },
  {
    title: 'Combine Updates to Improve the Shared Model',
    description: 'Your protected updates join with thousands of others to make a smarter model for everyone — still without ever revealing your data.',
    Icon: PublicIcon,
  },
  {
    title: 'Updated Model Comes Back to Your Device',
    description: 'The improved model is sent back to your phone — smarter, safer, and ready to learn again!',
    Icon: AutorenewIcon,
  },
];

function TrainingInfoGraphic() {
  const [currentSlide, setCurrentSlide] = useState(0);

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    arrows: true,
    swipe: true,
    autoplay: false,
    autoplaySpeed: 3000,
    beforeChange: (_, newIndex) => setCurrentSlide(newIndex),
    prevArrow: <PrevArrow />,
    nextArrow: <NextArrow />,
    adaptiveHeight: true,       
    centerMode: false,         
    centerPadding: '0px',      
  };
  

  return (
    <BaseButtonModal buttonText="What Happens During Training?" title="How Your Data Trains the Model">
      <div>
      <Typography variant="subtitle1" align="center" style={{fontWeight: "bold"}}>
        Step {currentSlide + 1} 
      </Typography>
        <Slider {...settings}>
          {steps.map((step, _) => {
            const IconComponent = step.Icon;
            return (
            <div 
                key={step.title}
              >
                <div style={{ textAlign: 'center', padding: '20px'}}>
                <IconComponent style={{ fontSize: 80, marginBottom: '20px', color: '#1976d2' }} />
                <Typography variant="h5" gutterBottom>{step.title}</Typography>
                <Typography variant="body1">{step.description}</Typography>
                </div>
              </div>
            );
          })}
        </Slider>
      </div>
    </BaseButtonModal>
  );
}

export default TrainingInfoGraphic;