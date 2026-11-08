import { NextResponse } from 'next/server';

// In-memory store for the MVP (can be upgraded to MongoDB/PostgreSQL later)
let latestAlert: any = {
  status: "SAFE",
  count: 0,
  zones: {},
  timestamp: new Date().toISOString()
};

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    
    // Log the incoming alert from the Python pipeline
    console.log('🚨 CrowdSafe Webhook Received:', payload);
    
    // Update the in-memory store
    latestAlert = {
      ...payload,
      timestamp: new Date().toISOString()
    };

    return NextResponse.json({ success: true, message: 'Webhook received successfully' }, { status: 200 });
  } catch (error) {
    console.error('Webhook payload error:', error);
    return NextResponse.json({ success: false, error: 'Invalid payload' }, { status: 400 });
  }
}

export async function GET() {
  // The React frontend will fetch this to update the UI
  return NextResponse.json({ data: latestAlert }, { status: 200 });
}