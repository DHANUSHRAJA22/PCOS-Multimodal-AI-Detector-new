import { useState } from 'react'
import { motion } from 'framer-motion'
import { TestTube, Download, Eye, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { toast } from 'sonner'
import { fixImageOrientation, type ProcessedImage } from '@/lib/image'

interface SampleImagesProps {
  onSelectSample: (type: 'face' | 'xray', processedImage: ProcessedImage) => void
}

const sampleImages = {
  face: [
    {
      id: 'face-sample-1',
      name: 'Sample Face 1',
      description: 'Clear frontal face photo - Normal indicators',
      url: 'https://as2.ftcdn.net/jpg/06/04/66/55/1000_F_604665506_enQwoqx6UAJkx6EeTcdDlBQEJT2pJhVc.jpg',
      expectedResult: 'Low Risk',
    },
    {
      id: 'face-sample-2',
      name: 'Sample Face 2',
      description: 'Professional headshot - Mixed indicators',
      url: 'https://static.wixstatic.com/media/11062b_4a9958c6480243868018fd9ed6b0bddd~mv2.jpg/v1/fill/w_1000,h_879,al_c,q_85,usm_0.66_1.00_0.01/11062b_4a9958c6480243868018fd9ed6b0bddd~mv2.jpg',
      expectedResult: 'Moderate Risk',
    },
  ],
  xray: [
    {
      id: 'xray-sample-1',
      name: 'Sample X-ray 1',
      description: 'Pelvic X-ray - Normal morphology',
      url: 'https://resources.ama.uk.com/glowm_www/uploads/1267016414_21d_Capture.JPG',
      expectedResult: 'Low Risk',
    },
    {
      id: 'xray-sample-2',
      name: 'Sample X-ray 2',
      description: 'Medical imaging - Complex patterns',
      url: 'https://www.emjreviews.com/wp-content/uploads/2022/06/Figure-2.jpg',
      expectedResult: 'High Risk',
    },
  ],
}

export function SampleImages({ onSelectSample }: SampleImagesProps) {
  const [isLoading, setIsLoading] = useState<string | null>(null)

  const choose = async (type: 'face' | 'xray', sample: { id: string; name: string; url: string }) => {
    setIsLoading(sample.id)
    try {
      // Route external URLs through our backend proxy (no local saving)
      const url = /^https?:\/\//i.test(sample.url)
        ? `/img-proxy?url=${encodeURIComponent(sample.url)}`
        : sample.url

      const res = await fetch(url)
      if (!res.ok) throw new Error(`Failed to fetch ${sample.name} (HTTP ${res.status})`)
      const blob = await res.blob()

      const file = new File([blob], `${sample.name.toLowerCase().replace(/\s+/g, '-')}.jpg`, {
        type: blob.type || 'image/jpeg',
        lastModified: Date.now(),
      })

      const processed = await fixImageOrientation(file)
      onSelectSample(type, processed)
      toast.success(`Loaded ${sample.name}`)
    } catch (e) {
      console.error(e)
      toast.error('Failed to load sample image')
    } finally {
      setIsLoading(null)
    }
  }

  const Row = (type: 'face' | 'xray') => (sample: any) => (
    <motion.div
      key={sample.id}
      whileHover={{ scale: 1.02 }}
      className="bg-white/70 p-3 rounded-lg border border-purple-200 hover:border-purple-300 transition-all"
    >
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => choose(type, sample)}
          className="flex items-center gap-3 flex-1 min-w-0 text-left"
          aria-label={`Use ${sample.name}`}
        >
          <img src={sample.url} alt={sample.name} className="w-12 h-12 rounded-lg object-cover" />
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm text-slate-800">{sample.name}</div>
            <div className="text-xs text-slate-600 truncate">{sample.description}</div>
            <Badge variant="outline" className="text-xs mt-1">
              Expected: {sample.expectedResult}
            </Badge>
          </div>
        </button>

        <div className="flex gap-1">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <Eye className="h-3 w-3" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>{sample.name}</DialogTitle>
              </DialogHeader>
              <img src={sample.url} alt={sample.name} className="w-full rounded-lg" />
              <p className="text-sm text-slate-600">{sample.description}</p>
            </DialogContent>
          </Dialog>

          <Button
            onClick={() => choose(type, sample)}
            disabled={isLoading === sample.id}
            size="sm"
            className="h-8 px-2 text-xs bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600"
            aria-label={`Load ${sample.name}`}
          >
            {isLoading === sample.id ? (
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" />
            ) : (
              <Download className="h-3 w-3" />
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  )

  return (
    <Card className="border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-indigo-50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <TestTube className="h-5 w-5 text-purple-600" />
          Try Sample Images
          <Badge className="ml-auto bg-gradient-to-r from-purple-500 to-indigo-500 text-white">
            <Sparkles className="h-3 w-3 mr-1" />
            Demo Mode
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        <p className="text-sm text-slate-600">
          Test the analyzer with pre-approved sample images. No personal data required.
        </p>

        <div className="space-y-3">
          <h4 className="font-semibold text-slate-800">Facial Analysis Samples</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sampleImages.face.map(Row('face'))}
          </div>
        </div>

        <div className="space-y-3">
          <h4 className="font-semibold text-slate-800">X-ray Analysis Samples</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sampleImages.xray.map(Row('xray'))}
          </div>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
          <p className="text-xs text-amber-800">
            <strong>Note:</strong> Sample images are for demonstration purposes only.
            Results may not reflect actual medical conditions.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
