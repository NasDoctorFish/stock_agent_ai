import * as QRCode from 'qrcode';
import { Request, Response } from 'express';
// 1. 이렇게 가져와야 'authenticator'를 찾을 수 있습니다.
import * as otplib from 'otplib';
// 2. 내부에서 꺼내서 이름을 붙여줍니다.
const { authenticator } = otplib as any;

// In a real application, this would be a user in your database
let userSecrets: Record<string, string> = {}; 

export const generateTotpSecret = async (req: Request, res: Response) => {
  const { userId, email } = req.body; // Assume userId and email are available

  // Generate a unique secret key
  const secret = authenticator.generateSecret();
  userSecrets[userId] = secret; // Store the secret securely

  // Generate the OTP Auth URI for the authenticator app
  const issuer = 'MyExpressApp'; // Your application name
  const label = email; // User identifier
  const otpauth = authenticator.generateURI({
    accountName: label,
    issuer,
    secret,
  });

  try {
    // Generate a QR code image URL (Data URL) from the URI
    const qrCodeImageUrl = await QRCode.toDataURL(otpauth);
    res.status(200).json({ 
      secret, 
      qrCodeUrl: qrCodeImageUrl,
      message: 'Scan the QR code with your authenticator app to enable 2FA.'
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate QR code' });
  }
};
