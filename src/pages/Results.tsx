import { useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Brain, ScanLine, CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ResultCard } from '@/components/ResultCard'
import { MedicalDisclaimer } from '@/components/MedicalDisclaimer'
import { RiskGauge } from '@/components/RiskGauge'
import { AIPerformanceMetrics } from '@/components/AIPerformanceMetrics'
import type { PredictionResponse } from '@/lib/api'

type RiskLevel = 'low' | 'moderate' | 'high' | 'unknown'

/** Case-insensitive whole-word check: prevents 'healthy' matching 'unhealthy'. */
function hasWord(haystack: string, word: string) {
  const re = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i')
  return re.test(haystack)
}

/** Normalize a risk string coming from backend if it ever exists. */
function normalizeRisk(r?: string | null): RiskLevel {
  const s = (r || '').trim().toLowerCase()
  if (s === 'low' || s === 'moderate' || s === 'high') return s
  return 'unknown'
}

/** Derive overall risk from the backend 'combined' sentence (current API). */
function riskFromCombined(combined?: string): RiskLevel {
  if (!combined) return 'unknown'
  const lower = combined.toLowerCase()
  const low = ['low risk', 'no pcos', 'non-pcos', 'negative', 'no symptoms']
  const high = ['high risk', 'positive', 'pcos detected', 'likely pcos']
  const mid  = ['moderate', 'possible', 'may suggest']
  if (low.some(w => lower.includes(w))) return 'low'
  if (high.some(w => lower.includes(w))) return 'high'
  if (mid.some(w => lower.includes(w))) return 'moderate'
  return 'unknown'
}

/** Map a single-modality prediction to Normal/Review with exact label handling. */
function summarizeSingle(prediction?: string, explicitRisk?: RiskLevel):
  { status: 'normal' | 'review'; text: string } {
  // Prefer explicit risk if backend provides it in the future
  if (explicitRisk && explicitRisk !== 'unknown') {
    return explicitRisk === 'low'
      ? { status: 'normal', text: 'Normal' }
      : { status: 'review', text: 'Review Needed' }
  }

  if (!prediction) return { status: 'review', text: 'No data' }
  const p = prediction.toLowerCase().trim()

  // Exact labels from your face model: ["non_pcos", "unhealthy"]
  if (p === 'non_pcos' || p === 'non-pcos') return { status: 'normal', text: 'Normal' }
  if (p === 'unhealthy' || p === 'pcos' || p === 'positive' || p === 'pcos_positive') {
    return { status: 'review', text: 'Review Needed' }
  }

  // Fallback for descriptive sentences (e.g., X-ray text)
  const normalWords = ['normal', 'healthy', 'no symptoms', 'negative', 'non-pcos', 'no pcos']
  if (normalWords.some(w => hasWord(p, w))) return { status: 'normal', text: 'Normal' }
  return { status: 'review', text: 'Review Needed' }
}

function riskExplanation(risk: RiskLevel): string {
  switch (risk) {
    case 'low':
      return 'The analysis shows minimal indicators typically associated with PCOS. Regular monitoring and healthy lifestyle practices are recommended.'
    case 'moderate':
      return 'The analysis shows some potential indicators that may warrant further investigation. Consider consulting a healthcare professional for comprehensive evaluation.'
    case 'high':
      return 'The analysis indicates several markers that could be associated with PCOS symptoms. Professional medical consultation is strongly recommended.'
    default:
      return 'The analysis results are inconclusive. Additional testing or consultation may be needed for proper evaluation.'
  }
}

const Reveal = ({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay }}>
    {children}
  </motion.div>
)

export function ResultsPage() {
  const { state } = useLocation() as { state?: { results?: PredictionResponse & Record<string, any> } }
  const navigate = useNavigate()
  const results = state?.results

  if (!results) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex items-center justify-center">
        <Card className="max-w-md mx-4 card-gradient shadow-vibrant-lg">
          <CardHeader className="text-center">
            <CardTitle className="text-gradient">No Results Found</CardTitle>
            <CardDescription>Please upload images first to see analysis results.</CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <Button 
              onClick={() => navigate('/')}
              className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-vibrant"
            >
              Return to Upload
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Prefer explicit risks if backend provides them later; otherwise parse 'combined'
  const overallRisk: RiskLevel =
    normalizeRisk((results as any).overall_risk) !== 'unknown'
      ? normalizeRisk((results as any).overall_risk)
      : riskFromCombined(results.combined)

  const faceRiskExplicit = normalizeRisk((results as any).face_risk)
  const xrayRiskExplicit = normalizeRisk((results as any).xray_risk)

  const faceSummary = summarizeSingle(results.face_pred, faceRiskExplicit)
  const xraySummary = summarizeSingle(results.xray_pred, xrayRiskExplicit)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-white/95 backdrop-blur-sm border-b border-purple-100">
        <div className="container mx-auto px-4 py-4">
          <Button 
            variant="ghost" 
            onClick={() => navigate('/')}
            className="font-medium hover:bg-purple-50 text-slate-700 hover:text-purple-700"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Upload
          </Button>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Overall Assessment */}
          <Reveal>
            <Card className="border-2 border-orange-200 bg-gradient-to-r from-orange-50 to-amber-50 shadow-vibrant-lg">
              <CardHeader>
                <CardTitle className="text-2xl flex items-center gap-3 text-slate-800">
                  <div className="p-3 rounded-lg bg-gradient-to-br from-orange-400 to-amber-400 text-white shadow-lg">
                    <ScanLine className="h-6 w-6" />
                  </div>
                  Overall Assessment
                </CardTitle>
              </CardHeader>

              <CardContent className="space-y-6">
                <div className="flex items-start gap-4">
                  <Badge 
                    className={`text-base px-6 py-3 capitalize font-semibold shadow-lg ${
                      overallRisk === 'low' 
                        ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600'
                        : overallRisk === 'high'
                        ? 'bg-gradient-to-r from-rose-500 to-red-500 text-white hover:from-rose-600 hover:to-red-600'
                        : 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-600 hover:to-orange-600'
                    }`}
                  >
                    <TrendingUp className="h-4 w-4 mr-2" />
                    {overallRisk === 'unknown' ? 'Pending Analysis' : `${overallRisk} Risk`}
                  </Badge>
                </div>

                <div className="bg-white/70 rounded-lg p-6 border border-orange-200">
                  <p className="text-lg leading-relaxed mb-4 text-slate-700">
                    {results.combined || 'Analysis in progress...'}
                  </p>
                </div>

                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
                  <h4 className="font-semibold mb-3 text-lg text-slate-800">Detailed Explanation</h4>
                  <p className="text-slate-600 leading-relaxed">
                    {riskExplanation(overallRisk)}
                  </p>
                </div>

                {/* Analysis Summary Chips */}
                <div className="flex flex-wrap gap-4">
                  {results.face_pred && (
                    <motion.div 
                      className={`flex items-center gap-3 px-4 py-3 rounded-full shadow-lg border-2 ${
                        faceSummary.status === 'normal' 
                          ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200' 
                          : 'bg-gradient-to-r from-orange-50 to-amber-50 border-orange-200'
                      }`}
                      whileHover={{ scale: 1.02 }}
                    >
                      {faceSummary.status === 'normal' ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                      )}
                      <span className="text-sm font-medium text-slate-700">
                        <span className="font-semibold">Facial Analysis:</span> {faceSummary.text}
                      </span>
                    </motion.div>
                  )}

                  {results.xray_pred && (
                    <motion.div 
                      className={`flex items-center gap-3 px-4 py-3 rounded-full shadow-lg border-2 ${
                        xraySummary.status === 'normal' 
                          ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200' 
                          : 'bg-gradient-to-r from-orange-50 to-amber-50 border-orange-200'
                      }`}
                      whileHover={{ scale: 1.02 }}
                    >
                      {xraySummary.status === 'normal' ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                      )}
                      <span className="text-sm font-medium text-slate-700">
                        <span className="font-semibold">Imaging Analysis:</span> {xraySummary.text}
                      </span>
                    </motion.div>
                  )}
                </div>
              </CardContent>
            </Card>
          </Reveal>

          {/* Risk Gauge Visualization */}
          <Reveal delay={0.05}>
            <div className="flex justify-center">
              <RiskGauge 
                riskLevel={overallRisk} 
                /* Using face confidence as a proxy if present */
                confidence={results.face_scores ? Math.max(...results.face_scores) * 100 : 0}
                className="max-w-sm"
              />
            </div>
          </Reveal>

          {/* Detailed Results */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {results.face_pred && (
              <Reveal delay={0.1}>
                <ResultCard
                  title="Facial Analysis"
                  prediction={results.face_pred}
                  scores={results.face_scores}
                  originalImage={results.face_img}
                  riskLevel={faceSummary.status === 'normal' ? 'low' : 'moderate'}
                />
              </Reveal>
            )}

            {results.xray_pred && (
              <Reveal delay={0.2}>
                <ResultCard
                  title="X-ray Analysis"
                  prediction={results.xray_pred}
                  originalImage={results.xray_img}
                  visualizationImage={results.yolo_vis}
                  foundLabels={results.found_labels}
                  riskLevel={xraySummary.status === 'normal' ? 'low' : 'moderate'}
                />
              </Reveal>
            )}
          </div>

          {/* AI Performance Metrics */}
          <Reveal delay={0.25}>
            <AIPerformanceMetrics />
          </Reveal>

          {/* Actions */}
          <Reveal delay={0.3}>
            <div className="text-center">
              <Button 
                onClick={() => navigate('/')}
                size="lg"
                className="px-12 py-6 text-lg bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-600 hover:from-purple-700 hover:via-indigo-700 hover:to-teal-700 shadow-vibrant-lg transition-all duration-300 hover:scale-105"
              >
                <Brain className="h-5 w-5 mr-2" />
                Analyze Another Image
              </Button>
            </div>
          </Reveal>

          {/* Medical Disclaimer */}
          <Reveal delay={0.4}>
            <MedicalDisclaimer />
          </Reveal>
        </div>
      </div>
    </div>
  )
}
