import { useState } from 'react';
import { Button, Dialog, DialogTitle, DialogContent } from '@mui/material';

function BaseButtonModal({ buttonText, title, children }) {
  const [open, setOpen] = useState(false);

  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <>
      <Button variant="outlined" color="secondary" onClick={handleOpen} style={{ margin: '0 0 20px 0' }}>
        {buttonText}
      </Button>

      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="md"
        fullWidth
        scroll="body"
        disableScrollLock={true}
        height="100%"
      >
        <DialogTitle>{title}</DialogTitle>
        <DialogContent dividers style={{ overflow: 'hidden', height: '100%', padding: '20px 20px 40px 20px' }}>
          {children}
        </DialogContent>
      </Dialog>
    </>
  );
}

export default BaseButtonModal;
