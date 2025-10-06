// server.js - MindEase Backend with BERT Integration
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/mindease', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('✅ MongoDB Connected'))
.catch(err => console.error('❌ MongoDB Connection Error:', err));

// ============================================
// MODELS
// ============================================

// User Model
const userSchema = new mongoose.Schema({
  username: { type: String, unique: true, sparse: true },
  email: { type: String, unique: true, sparse: true },
  password: { type: String },
  isAnonymous: { type: Boolean, default: false },
  anonymousId: { type: String, unique: true, sparse: true },
  createdAt: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);

// Mood Entry Model
const moodSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  mood: { type: String, required: true },
  intensity: { type: Number, min: 1, max: 10, required: true },
  note: { type: String, maxlength: 500 },
  activities: [String],
  aiAnalysis: {
    sentiment: String,
    confidence: Number,
    detectedEmotions: [String],
    riskLevel: String,
    insights: [String],
    emotionScores: Array
  },
  timestamp: { type: Date, default: Date.now }
});

const MoodEntry = mongoose.model('MoodEntry', moodSchema);

// Community Post Model
const postSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  content: { type: String, required: true, maxlength: 1000 },
  isAnonymous: { type: Boolean, default: true },
  likes: { type: Number, default: 0 },
  supportCount: { type: Number, default: 0 },
  comments: [{
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    content: String,
    isAnonymous: Boolean,
    timestamp: { type: Date, default: Date.now }
  }],
  createdAt: { type: Date, default: Date.now }
});

const Post = mongoose.model('Post', postSchema);

// Coping Strategy Model
const copingStrategySchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  strategy: { type: String, required: true },
  category: { type: String },
  isAIGenerated: { type: Boolean, default: false },
  effectiveness: { type: Number, min: 1, max: 5 },
  usedCount: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now }
});

const CopingStrategy = mongoose.model('CopingStrategy', copingStrategySchema);

// ============================================
// AI SERVICE LAYER (BERT Integration)
// ============================================
class AIService {
  constructor() {
    this.aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
  }

  async analyzeSentiment(text) {
    try {
      const response = await fetch(`${this.aiServiceUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`AI Service error: ${response.statusText}`);
      }

      const data = await response.json();
      
      return {
        sentiment: data.sentiment,
        confidence: data.confidence,
        detectedEmotions: data.detectedEmotions || [],
        riskLevel: data.riskLevel,
        insights: data.insights || [],
        emotionScores: data.emotionScores || []
      };
    } catch (error) {
      console.error('⚠️ BERT Analysis Error:', error.message);
      return this.fallbackSentimentAnalysis(text);
    }
  }

  fallbackSentimentAnalysis(text) {
    const keywords = {
      positive: ['happy', 'great', 'good', 'excellent', 'wonderful', 'joy', 'excited', 'love'],
      negative: ['sad', 'bad', 'terrible', 'awful', 'depressed', 'anxious', 'worried', 'stressed', 'hate', 'angry'],
    };

    const lowerText = text.toLowerCase();
    let score = 0;
    let detectedEmotions = [];

    keywords.positive.forEach(word => {
      if (lowerText.includes(word)) {
        score += 1;
        detectedEmotions.push(word);
      }
    });
    keywords.negative.forEach(word => {
      if (lowerText.includes(word)) {
        score -= 1;
        detectedEmotions.push(word);
      }
    });

    let sentiment, riskLevel;
    if (score > 0) {
      sentiment = 'positive';
      riskLevel = 'low';
    } else if (score < -2) {
      sentiment = 'negative';
      riskLevel = 'high';
    } else if (score < 0) {
      sentiment = 'negative';
      riskLevel = 'medium';
    } else {
      sentiment = 'neutral';
      riskLevel = 'low';
    }

    return {
      sentiment,
      confidence: 0.5,
      detectedEmotions: [...new Set(detectedEmotions)],
      riskLevel,
      fallback: true,
      insights: ['Using fallback analysis. BERT service unavailable.']
    };
  }

  async generateCopingSuggestions(moodData, userHistory = []) {
    const suggestions = {
      anxious: [
        { strategy: '4-7-8 Breathing Exercise', category: 'breathing', description: 'Breathe in for 4, hold for 7, out for 8' },
        { strategy: 'Progressive Muscle Relaxation', category: 'relaxation', description: 'Tense and release each muscle group' },
        { strategy: 'Grounding Exercise (5-4-3-2-1)', category: 'mindfulness', description: 'Name 5 things you see, 4 you touch...' }
      ],
      sad: [
        { strategy: 'Gratitude Journaling', category: 'journaling', description: 'Write 3 things you\'re grateful for' },
        { strategy: 'Physical Activity', category: 'exercise', description: 'Take a 10-minute walk outside' },
        { strategy: 'Connect with Someone', category: 'social', description: 'Reach out to a friend or family member' }
      ],
      stressed: [
        { strategy: 'Pomodoro Break', category: 'productivity', description: 'Take a 5-minute break every 25 minutes' },
        { strategy: 'Guided Meditation', category: 'meditation', description: 'Try a 10-minute meditation session' },
        { strategy: 'Time Blocking', category: 'planning', description: 'Organize tasks into manageable blocks' }
      ],
      happy: [
        { strategy: 'Mood Boosting Activity', category: 'engagement', description: 'Do something creative or fun' },
        { strategy: 'Share Positivity', category: 'social', description: 'Share your good mood with the community' }
      ],
      neutral: [
        { strategy: 'Mindful Check-in', category: 'mindfulness', description: 'Take a moment to assess your needs' },
        { strategy: 'Set an Intention', category: 'planning', description: 'Set a positive intention for the day' }
      ]
    };

    const mood = moodData.mood.toLowerCase();
    return suggestions[mood] || suggestions.neutral;
  }

  async analyzeUserPatterns(userId) {
    const recentEntries = await MoodEntry.find({ userId })
      .sort({ timestamp: -1 })
      .limit(30);

    if (recentEntries.length === 0) {
      return { message: 'Not enough data yet' };
    }

    const moodCounts = {};
    const avgIntensity = recentEntries.reduce((sum, entry) => {
      moodCounts[entry.mood] = (moodCounts[entry.mood] || 0) + 1;
      return sum + entry.intensity;
    }, 0) / recentEntries.length;

    const dominantMood = Object.keys(moodCounts).reduce((a, b) => 
      moodCounts[a] > moodCounts[b] ? a : b
    );

    return {
      averageIntensity: avgIntensity.toFixed(2),
      dominantMood,
      moodDistribution: moodCounts,
      totalEntries: recentEntries.length,
      insight: this.generateInsight(dominantMood, avgIntensity)
    };
  }

  generateInsight(mood, intensity) {
    if (mood === 'anxious' || mood === 'stressed') {
      if (intensity > 7) {
        return 'Your stress levels have been high. Consider talking to someone or trying relaxation exercises.';
      }
      return 'You\'ve been experiencing some stress. Regular breaks and self-care can help.';
    }
    if (mood === 'sad' && intensity > 6) {
      return 'You\'ve been feeling down lately. Remember, it\'s okay to reach out for support.';
    }
    if (mood === 'happy') {
      return 'Great to see positive moods! Keep doing what works for you.';
    }
    return 'Your mood has been fairly balanced. Keep tracking to maintain awareness.';
  }
}

const aiService = new AIService();

// ============================================
// MIDDLEWARE
// ============================================
const authMiddleware = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    const user = await User.findById(decoded.userId);

    if (!user) {
      return res.status(401).json({ error: 'User not found' });
    }

    req.user = user;
    req.userId = user._id;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid authentication' });
  }
};

// ============================================
// AUTH ROUTES
// ============================================

app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, password, username } = req.body;

    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({
      email,
      username,
      password: hashedPassword,
      isAnonymous: false
    });

    await user.save();

    const token = jwt.sign({ userId: user._id }, JWT_SECRET, { expiresIn: '30d' });

    res.status(201).json({
      message: 'User registered successfully',
      token,
      user: { id: user._id, email: user.email, username: user.username }
    });
  } catch (error) {
    res.status(500).json({ error: 'Registration failed', details: error.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: user._id }, JWT_SECRET, { expiresIn: '30d' });

    res.json({
      message: 'Login successful',
      token,
      user: { id: user._id, email: user.email, username: user.username }
    });
  } catch (error) {
    res.status(500).json({ error: 'Login failed', details: error.message });
  }
});

app.post('/api/auth/anonymous', async (req, res) => {
  try {
    const anonymousId = `anon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const user = new User({
      isAnonymous: true,
      anonymousId
    });

    await user.save();

    const token = jwt.sign({ userId: user._id }, JWT_SECRET, { expiresIn: '30d' });

    res.status(201).json({
      message: 'Anonymous session created',
      token,
      user: { id: user._id, anonymousId, isAnonymous: true }
    });
  } catch (error) {
    res.status(500).json({ error: 'Anonymous login failed', details: error.message });
  }
});

// ============================================
// MOOD TRACKING ROUTES
// ============================================

app.post('/api/moods', authMiddleware, async (req, res) => {
  try {
    const { mood, intensity, note, activities } = req.body;

    let aiAnalysis = null;
    if (note) {
      aiAnalysis = await aiService.analyzeSentiment(note);
    }

    const moodEntry = new MoodEntry({
      userId: req.userId,
      mood,
      intensity,
      note,
      activities: activities || [],
      aiAnalysis
    });

    await moodEntry.save();

    const suggestions = await aiService.generateCopingSuggestions(moodEntry);

    res.status(201).json({
      message: 'Mood logged successfully',
      moodEntry,
      suggestions,
      analysis: aiAnalysis
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to log mood', details: error.message });
  }
});

app.get('/api/moods', authMiddleware, async (req, res) => {
  try {
    const { limit = 30, days = 30 } = req.query;
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));

    const moods = await MoodEntry.find({
      userId: req.userId,
      timestamp: { $gte: startDate }
    })
    .sort({ timestamp: -1 })
    .limit(parseInt(limit));

    res.json({ moods, count: moods.length });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch moods', details: error.message });
  }
});

app.get('/api/moods/insights', authMiddleware, async (req, res) => {
  try {
    const insights = await aiService.analyzeUserPatterns(req.userId);
    res.json({ insights });
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate insights', details: error.message });
  }
});

// ============================================
// COPING STRATEGIES ROUTES
// ============================================

app.get('/api/coping/suggestions', authMiddleware, async (req, res) => {
  try {
    const { mood } = req.query;
    
    const recentMoods = await MoodEntry.find({ userId: req.userId })
      .sort({ timestamp: -1 })
      .limit(10);

    const moodData = { mood: mood || 'neutral' };
    const suggestions = await aiService.generateCopingSuggestions(moodData, recentMoods);

    res.json({ suggestions });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get suggestions', details: error.message });
  }
});

app.post('/api/coping/strategies', authMiddleware, async (req, res) => {
  try {
    const { strategy, category, isAIGenerated } = req.body;

    const copingStrategy = new CopingStrategy({
      userId: req.userId,
      strategy,
      category,
      isAIGenerated: isAIGenerated || false
    });

    await copingStrategy.save();
    res.status(201).json({ message: 'Strategy saved', copingStrategy });
  } catch (error) {
    res.status(500).json({ error: 'Failed to save strategy', details: error.message });
  }
});

app.patch('/api/coping/strategies/:id/rate', authMiddleware, async (req, res) => {
  try {
    const { effectiveness } = req.body;
    
    const strategy = await CopingStrategy.findOneAndUpdate(
      { _id: req.params.id, userId: req.userId },
      { 
        effectiveness,
        $inc: { usedCount: 1 }
      },
      { new: true }
    );

    if (!strategy) {
      return res.status(404).json({ error: 'Strategy not found' });
    }

    res.json({ message: 'Rating saved', strategy });
  } catch (error) {
    res.status(500).json({ error: 'Failed to rate strategy', details: error.message });
  }
});

// ============================================
// COMMUNITY ROUTES
// ============================================

app.post('/api/community/posts', authMiddleware, async (req, res) => {
  try {
    const { content, isAnonymous } = req.body;

    const post = new Post({
      userId: req.userId,
      content,
      isAnonymous: isAnonymous !== undefined ? isAnonymous : true
    });

    await post.save();
    res.status(201).json({ message: 'Post created', post });
  } catch (error) {
    res.status(500).json({ error: 'Failed to create post', details: error.message });
  }
});

app.get('/api/community/posts', authMiddleware, async (req, res) => {
  try {
    const { limit = 20, skip = 0 } = req.query;

    const posts = await Post.find()
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip))
      .select('-userId');

    res.json({ posts, count: posts.length });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch posts', details: error.message });
  }
});

app.post('/api/community/posts/:id/comments', authMiddleware, async (req, res) => {
  try {
    const { content, isAnonymous } = req.body;

    const post = await Post.findById(req.params.id);
    if (!post) {
      return res.status(404).json({ error: 'Post not found' });
    }

    post.comments.push({
      userId: req.userId,
      content,
      isAnonymous: isAnonymous !== undefined ? isAnonymous : true
    });

    await post.save();
    res.json({ message: 'Comment added', post });
  } catch (error) {
    res.status(500).json({ error: 'Failed to add comment', details: error.message });
  }
});

app.post('/api/community/posts/:id/support', authMiddleware, async (req, res) => {
  try {
    const post = await Post.findByIdAndUpdate(
      req.params.id,
      { $inc: { supportCount: 1 } },
      { new: true }
    );

    if (!post) {
      return res.status(404).json({ error: 'Post not found' });
    }

    res.json({ message: 'Support added', post });
  } catch (error) {
    res.status(500).json({ error: 'Failed to add support', details: error.message });
  }
});

// ============================================
// HEALTH CHECK
// ============================================
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'MindEase API is running',
    timestamp: new Date().toISOString()
  });
});

// ============================================
// START SERVER
// ============================================
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('🚀 MindEase API Server');
  console.log('='.repeat(60));
  console.log(`📡 Server running on http://localhost:${PORT}`);
  console.log(`📊 MongoDB: ${mongoose.connection.readyState === 1 ? '✅ Connected' : '⏳ Connecting...'}`);
  console.log(`🤖 AI Service: ${process.env.AI_SERVICE_URL || 'http://localhost:8000'}`);
  console.log('='.repeat(60));
});